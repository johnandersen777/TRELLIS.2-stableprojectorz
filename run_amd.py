#!/usr/bin/env python
"""
TRELLIS.2 image-to-3D runner — AMD ROCm edition.

Handles all CUDA extension fallbacks:
  - o-voxel._C → pure-PyTorch hashmap (hashmap_fallback.py)
  - flex_gemm → dummy module (not used in pipeline, only in postprocess)
  - cumesh, nvdiffrast → dummy modules (not used in pipeline)
  - SparseConv → pure-PyTorch (conv_none.py)
  - GLB export → pure trimesh (export_glb_fallback.py)
"""

import os
import sys

# ── Environment setup ──────────────────────────────────────────────
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

# ── Monkey-patch CUDA-only modules before anything imports them ───
import importlib
import importlib.machinery
import types
import warnings

# Helper: create dummy module that raises on access
def _make_dummy_module(name, real_import=None):
    """Return a dummy module. If real_import is provided, try that first."""
    if real_import is not None:
        try:
            return importlib.import_module(real_import)
        except ImportError:
            pass
    mod = types.ModuleType(name)
    mod.__doc__ = f"Dummy replacement for {name} (GPU extension not available)"
    # Set __spec__ to prevent importlib.util.find_spec from crashing
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = [name]  # Make it look like a package
    mod.__file__ = f"<dummy:{name}>"
    return mod

# ── Build proper dummy module tree for flex_gemm ────────────────────
# flex_gemm.ops.spconv is imported at module level in conv_flex_gemm.py
# We need: flex_gemm → flex_gemm.ops → flex_gemm.ops.spconv
# with all the symbols that conv_flex_gemm expects

# Create flex_gemm package
_flex_gemm = _make_dummy_module("flex_gemm")
sys.modules["flex_gemm"] = _flex_gemm

# Create flex_gemm.ops subpackage
_flex_gemm_ops = _make_dummy_module("flex_gemm.ops")
_flex_gemm.ops = _flex_gemm_ops
sys.modules["flex_gemm.ops"] = _flex_gemm_ops

# Create flex_gemm.ops.spconv with needed symbols
_flex_gemm_spconv = _make_dummy_module("flex_gemm.ops.spconv")

# Enum-like Algorithm class expected by conv_flex_gemm
class Algorithm:
    EXPLICIT_GEMM = 0
    IMPLICIT_GEMM = 1
    MASKED_IMPLICIT_GEMM = 2
    MASKED_IMPLICIT_GEMM_SPLITK = 3

_flex_gemm_spconv.Algorithm = Algorithm
_flex_gemm_spconv.sparse_submanifold_conv3d = lambda *a, **kw: None
_flex_gemm_spconv.SubMConv3dFunction = type("SubMConv3dFunction", (), {
    "_compute_neighbor_cache": staticmethod(lambda *a, **kw: {}),
})()
_flex_gemm_spconv.set_algorithm = lambda a: None
_flex_gemm_spconv.set_hashmap_ratio = lambda r: None

_flex_gemm_ops.spconv = _flex_gemm_spconv
sys.modules["flex_gemm.ops.spconv"] = _flex_gemm_spconv

# Create flex_gemm.ops.grid_sample (used by postprocess, not in pipeline)
_flex_gemm_grid = _make_dummy_module("flex_gemm.ops.grid_sample")
_flex_gemm_grid.grid_sample_3d = lambda *a, **kw: None
_flex_gemm_ops.grid_sample = _flex_gemm_grid
sys.modules["flex_gemm.ops.grid_sample"] = _flex_gemm_grid

# Register other dummy modules with proper __spec__
for dummy_name in [
    "nvdiffrast", "nvdiffrast.torch", "nvdiffrec_render",
    "cumesh", "flash_attn", "flash_attn_interface",
    "xformers", "xformers.ops",
]:
    if dummy_name not in sys.modules:
        m = _make_dummy_module(dummy_name)
        sys.modules[dummy_name] = m

# Patch cumesh with dummy CuMesh pass-through (AMD fallback: no CUDA mesh simplify)
class _DummyCuMesh:
    def init(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
    def remove_faces(self, face_mask):
        keep = face_mask.bool() if not hasattr(face_mask, 'dtype') else face_mask
        self.faces = self.faces[keep]
    def simplify(self, target=1000000, verbose=False, options=None):
        pass  # no-op: mesh already below target, trimesh decimation used instead
    def read(self):
        return self.vertices, self.faces
sys.modules["cumesh"].CuMesh = _DummyCuMesh

# Build nvdiffrast.torch sub-structure
_nvd = sys.modules["nvdiffrast"]
_nvd.torch = _make_dummy_module("nvdiffrast.torch")
sys.modules["nvdiffrast.torch"] = _nvd.torch

# ── Load o_voxel Python sources with fallback hashmap _C ───────────
from hashmap_fallback import hashmap_insert_3d_cuda, hashmap_lookup_3d_cuda

# Create _C with hashmap functions — MUST register BEFORE importing o_voxel
# because flexible_dual_grid.py does `from .. import _C` at module level
_oc = types.ModuleType("o_voxel._C")
_oc.__spec__ = importlib.machinery.ModuleSpec("o_voxel._C", None)
_oc.hashmap_insert_3d_idx_as_val_cuda = lambda *a: hashmap_insert_3d_cuda(*a[:2], a[2], *a[3:])
_oc.hashmap_lookup_3d_cuda = lambda *a: hashmap_lookup_3d_cuda(*a[:2], a[2], *a[3:])
_oc.mesh_to_flexible_dual_grid_cpu = lambda *a: (_ for _ in ()).throw(
    NotImplementedError("mesh_to_flexible_dual_grid_cpu not available"))

# Remove any stale dummies
for _mod in list(sys.modules.keys()):
    if _mod.startswith("o_voxel") or _mod.startswith("o-voxel"):
        del sys.modules[_mod]

# Pre-register _C before o_voxel package init
sys.modules["o_voxel._C"] = _oc

_o_voxel_path = os.path.join(os.path.dirname(__file__), "o-voxel", "o_voxel")
if os.path.isdir(_o_voxel_path):
    sys.path.insert(0, os.path.dirname(_o_voxel_path))
    import o_voxel
    import o_voxel.convert
    import o_voxel.convert.flexible_dual_grid
    import o_voxel.postprocess
    # Patch _C references
    o_voxel._C = _oc
    o_voxel.convert._C = _oc
    o_voxel.convert.flexible_dual_grid._C = _oc
    o_voxel.postprocess._C = _oc
    print("[OK] o_voxel Python modules loaded with fallback _C")
else:
    print(f"[!] o-voxel source not found at {_o_voxel_path}")

print("[AMD] CUDA extension fallbacks registered")

# ── GPU detection ──────────────────────────────────────────────────
import torch
print(f"PyTorch {torch.__version__}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_mem/1024**3:.1f} GB total ({free_mem/1024**3:.1f} GB free)")
    DEVICE = "cuda"
    # AMD-specific HIP config
    if "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper():
        os.environ["ATTN_BACKEND"] = "sdpa"
        print("[AMD] Radeon GPU detected — using SDPA attention, conv_none backend")
else:
    print("[!] No GPU detected, falling back to CPU")
    DEVICE = "cpu"

# ── Now import the pipeline ────────────────────────────────────────
print("[*] Loading TRELLIS.2 pipeline...")

# Monkey-patch o_voxel._C with fallback before o_voxel imports it
from hashmap_fallback import (
    hashmap_insert_3d_cuda,
    hashmap_lookup_3d_cuda,
    hashmap_insert_3d_cpu,
    hashmap_lookup_3d_cpu,
)

# Create fake _C module for o_voxel
_oc = types.ModuleType("o_voxel._C")
_oc.hashmap_insert_3d_idx_as_val_cuda = lambda *args: hashmap_insert_3d_cuda(*args[:2], args[2], *args[3:])
_oc.hashmap_lookup_3d_cuda = lambda *args: hashmap_lookup_3d_cuda(*args[:2], args[2], *args[3:])
_oc.mesh_to_flexible_dual_grid_cpu = lambda *args: (_ for _ in ()).throw(NotImplementedError("mesh_to_flexible_dual_grid_cpu not available"))
sys.modules["o_voxel._C"] = _oc

# Patch o_voxel modules to use our _C
try:
    import o_voxel
    import o_voxel.convert
    import o_voxel.convert.flexible_dual_grid
    o_voxel.convert.flexible_dual_grid._C = _oc
    o_voxel._C = _oc
    print("[OK] o_voxel modules patched with fallback hashmap")
except ImportError as e:
    print(f"[!] Could not import o_voxel: {e}")
    # Create minimal o_voxel package
    if "o_voxel" not in sys.modules:
        _ov = types.ModuleType("o_voxel")
        _ov._C = _oc
        sys.modules["o_voxel"] = _ov
    if "o_voxel.convert" not in sys.modules:
        _ovc = types.ModuleType("o_voxel.convert")
        _ovc._C = _oc
        sys.modules["o_voxel.convert"] = _ovc
        sys.modules["o_voxel"].convert = _ovc

# Ensure trellis2 can import conv_flex_gemm (dummy)
if "trellis2.modules.sparse.conv" not in sys.modules:
    # Will be loaded when pipeline imports
    pass

import trellis2
# Patch the conv module's conv_flex_gemm reference
conv_mod = importlib.import_module("trellis2.modules.sparse.conv")
if not hasattr(conv_mod, "conv_flex_gemm"):
    conv_flex_gemm = types.ModuleType("trellis2.modules.sparse.conv.conv_flex_gemm")
    conv_flex_gemm.sparse_conv3d_init = lambda *a, **k: None
    conv_flex_gemm.sparse_conv3d_forward = lambda *a, **k: None
    conv_flex_gemm.sparse_inverse_conv3d_init = lambda *a, **k: None
    conv_flex_gemm.sparse_inverse_conv3d_forward = lambda *a, **k: None
    conv_mod.conv_flex_gemm = conv_flex_gemm

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from PIL import Image

# ── Main ───────────────────────────────────────────────────────────
import argparse


def main():
    ap = argparse.ArgumentParser(description="TRELLIS.2 AMD ROCm runner")
    ap.add_argument("input", help="Input image (png/jpg)")
    ap.add_argument("output", help="Output .glb file")
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--pipeline-type", default="1024_cascade",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="Skip mesh cache, always regenerate")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    device = "cpu" if args.cpu else DEVICE

    # ── Mesh cache: pickle file to skip pipeline on re-runs ──────────
    import hashlib
    import pickle
    _cache_path = args.output + ".cache.pkl"
    _cache_key = hashlib.md5(
        f"{os.path.abspath(args.input)}|{args.pipeline_type}|{args.seed}|{args.model}".encode()
    ).hexdigest()

    if not args.no_cache and os.path.exists(_cache_path):
        try:
            with open(_cache_path, "rb") as f:
                _cached = pickle.load(f)
            if _cached.get("key") == _cache_key and "vertices" in _cached:
                print(f"[cache] Loaded mesh from {_cache_path} ({_cached['vertices'].shape[0]} verts)")
                # Reconstruct minimal mesh-like object for export
                class _CachedMesh:
                    pass
                mesh = _CachedMesh()
                mesh.vertices = _cached["vertices"]
                mesh.faces = _cached["faces"]
                mesh.attrs = _cached.get("attrs")
                mesh.coords = _cached.get("coords")
                mesh.layout = _cached.get("layout")
                # Skip pipeline entirely — go straight to export
                _skip_pipeline = True
            else:
                print("[cache] Cache key mismatch, regenerating...")
                _skip_pipeline = False
        except Exception as e:
            print(f"[cache] Failed to load cache: {e}")
            _skip_pipeline = False
    else:
        _skip_pipeline = False

    if _skip_pipeline:
        print("[*] Skipping pipeline (cached mesh loaded)")
    else:
        print(f"[*] Loading pipeline: {args.model}")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
        pipeline.to(device)
        pipeline.low_vram = True
        print("[*] Pipeline loaded")

        print(f"[*] Loading image: {args.input}")
        image = Image.open(args.input)
        print(f"    Size: {image.size}, Mode: {image.mode}")

        print(f"[*] Generating 3D mesh (pipeline={args.pipeline_type}, seed={args.seed})...")
        print("    This may take 5-15 minutes on first run (model download + inference)")

        mesh = pipeline.run(
            image,
            num_samples=1,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
        )[0]

        print(f"[*] Mesh generated: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")

        # Simplify mesh
        mesh.simplify(16777216)
        print(f"[*] After simplify: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")

        # ── Save to pickle cache ─────────────────────────────────────
        if not args.no_cache:
            try:
                _data = {
                    "key": _cache_key,
                    "vertices": mesh.vertices.cpu(),
                    "faces": mesh.faces.cpu(),
                }
                if hasattr(mesh, 'attrs') and mesh.attrs is not None:
                    _data["attrs"] = mesh.attrs.cpu()
                if hasattr(mesh, 'coords') and mesh.coords is not None:
                    _data["coords"] = mesh.coords.cpu()
                if hasattr(mesh, 'layout'):
                    _data["layout"] = mesh.layout
                with open(_cache_path, "wb") as f:
                    pickle.dump(_data, f)
                print(f"[*] Cached mesh to {_cache_path}")
            except Exception as e:
                print(f"[!] Cache write failed: {e}")

    # Export using fallback (no CUDA extensions needed)
    print(f"[*] Exporting to GLB: {args.output}")
    from export_glb_fallback import mesh_to_glb

    if hasattr(mesh, 'attrs') and hasattr(mesh, 'coords'):
        trimesh_obj = mesh_to_glb(
            mesh.vertices, mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            verbose=args.verbose,
        )
    else:
        trimesh_obj = mesh_to_glb(
            mesh.vertices, mesh.faces,
            verbose=args.verbose,
        )
    trimesh_obj.export(args.output)
    print(f"[done] Saved {args.output}")


if __name__ == "__main__":
    main()
