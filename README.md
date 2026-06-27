# TRELLIS.2 image-to-3D on AMD Radeon RX 9070 XT (Windows 11, ROCm)

Run the [TRELLIS.2](https://github.com/IgorAherne/TRELLIS.2-stableprojectorz) image→3D
pipeline on an AMD GPU (no NVIDIA/CUDA), producing a clean watertight textured `.glb`.

**Status: working end-to-end.** Verified on two vehicles from a single photo each:
- `reference-images/T-80BVM.jpg` (T-80 tank, pulled from Wikipedia) → `T-80BVM.glb`
- `reference-images/Humvee.jpg` (Humvee, pulled from Wikipedia) → `Humvee.glb`

Both come out watertight, vertex-colored, and recognizable from every angle.

<img width="1034" height="1041" alt="humve" src="https://github.com/user-attachments/assets/5ff28440-617b-47b2-aad3-3ded7ea88e11" />

<img width="1193" height="975" alt="t-80" src="https://github.com/user-attachments/assets/8e50684d-ad7f-48dc-9e33-15c4227e0ecb" />

## Hardware / environment
- **GPU**: AMD Radeon RX 9070 XT (RDNA 4, `gfx1201`), 15.9 GB VRAM — no NVIDIA GPU
- **System RAM**: 31 GB (pipeline peaks ~19 GB during model load, ~15 GB steady)
- **OS**: Windows 11 + Git Bash + PowerShell
- **Stack**: ROCm 7.2.1, PyTorch 2.9.1+rocm7.2.1, Python 3.12, in `TRELLIS.2-stableprojectorz/venv/`

## Quick start
```bash
cd TRELLIS.2-stableprojectorz
source venv/Scripts/activate
HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
SPARSE_CONV_BACKEND=none \
SPARSE_ATTN_BACKEND=sdpa \
ATTN_BACKEND=sdpa \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -u run_amd.py \
  ../reference-images/T-80BVM.jpg \
  ../reference-images/T-80BVM.glb \
  --pipeline-type 512
```
- `--no-cache` forces a full regenerate (otherwise a `*.glb.cache.pkl` of the raw mesh is
  reused, so you can iterate on the exporter without re-running inference).
- Runtime: ~2 min after models are cached. First run downloads ~15 GB of checkpoints.

### View the result
```bash
python -m http.server 8080   # from the repo root (trellis.2/)
# open: http://localhost:8080/glb-viewer.html?model=reference-images/T-80BVM.glb
```
`glb-viewer.html` uses Google `<model-viewer>` (supports the `?model=` URL param,
drag-and-drop, and an Open button).

## What made it work

Getting from "loads but outputs garbage" to "clean textured vehicle" came down to three
fixes. The debugging deliberately *proved* several suspects innocent first.

### 1. The garbage-geometry bug — SDPA attention channel permute
`trellis2/modules/sparse/attention/full_attn.py`, the `sdpa` fallback branch.

The output reshape used `permute(1, 2, 0)` → `(L, C, H)` → `(L, C*H)` **C-major**, while the
reference flash_attn/xformers path returns `(L, H, C)` **H-major**. For multi-head attention
(H>1) this **permutes the feature channels at every attention layer**, corrupting the
flow-model latents → salt-and-pepper voxel occupancy (7180 disconnected blobs) → shredded
mesh (32k components, largest only 4.3%).

**Fix:** `permute(1, 2, 0)` → `permute(1, 0, 2)`. Verified numerically against a reference
multi-head attention: max error dropped from **2.47 → 5e-7**. After the fix the mesh became a
single connected tank (largest component 83.9% of faces).

Proven NOT the cause (each tested directly):
- `hashmap_fallback.py` — 100% correct self+neighbor lookups vs brute force
- `conv_none.py` sparse conv — matches dense `F.conv3d` to 1e-5
- mesh-extraction sentinel handling — torch wraps the scalar to int32, works
- the viewer — `<model-viewer>` (three.js GLTFLoader) rendered identical scatter

### 2. The porous-surface problem — remesh from occupancy
The raw dual-contour output is porous (~409k open boundary edges) because the real CUDA
`to_glb` does BVH remeshing that the pure-Python fallback can't replicate. The **voxel
occupancy is complete**, though, so `export_glb_fallback.py` reconstructs from it instead:
scatter `coords` into a dense grid → binary-close 1-voxel gaps → gaussian → `skimage`
marching cubes → **watertight by construction**. Keep large components, Taubin-smooth.

### 3. The rainbow-texture problem — symmetry color mirroring
Per-vertex `base_color` comes from the pipeline's `attrs`. The occluded side of a single-view
input has iridescent "rainbow" texture noise that is *locally* smooth, so per-voxel variance
or alpha filtering can't catch it. The exporter instead exploits **bilateral symmetry**:
1. find the width/symmetry axis via occupancy mirror-IoU,
2. compare chroma-hue variance of the two halves (clean ≈ 0.001 vs noisy ≈ 0.08–0.14),
3. mirror the clean half's color onto the noisy half,
4. EDT nearest-color fill + gaussian smooth + a few Laplacian mesh-graph passes.

Auto-guarded: only mirrors when one half is clearly noisier (variance ratio > 3×), otherwise
uses all voxels as the color source. This generalized unchanged from the tank to the Humvee
(auto-detected `sym_axis=0 mirror=on` for both).

## Fallback modules (this port)
| File | Role |
|------|------|
| `run_amd.py` | AMD runner. Registers dummy CUDA-only modules (flex_gemm, nvdiffrast, cumesh, flash_attn, xformers), wires the fallback `o_voxel._C`, sets ROCm env, pickle mesh cache. |
| `trellis2/modules/sparse/conv/conv_none.py` | Pure-PyTorch submanifold sparse conv (`SPARSE_CONV_BACKEND=none`). Weight layout `(Co,Kd,Kh,Kw,Ci)` matches the checkpoint. |
| `trellis2/modules/sparse/attention/full_attn.py` | `sdpa` branch for variable-length sparse attention (the channel-permute fix lives here). |
| `hashmap_fallback.py` | GPU-sorted `flat_idx` hashmap via `torch.sort` + `torch.searchsorted` — replaces o-voxel `_C` hashmap. |
| `export_glb_fallback.py` | Occupancy marching-cubes remesh + symmetry-aware vertex colors → watertight textured GLB. |
| `glb-viewer.html` | `<model-viewer>`-based WebGPU viewer with `?model=` auto-load. |
| `setup.sh` | ROCm SDK + PyTorch + deps install. |

Extra Python deps beyond the base install: `scikit-image`, `scipy` (remesh + color fill).
`pymeshfix` was tried for watertight repair but was too slow on multi-million-face meshes;
not used in the final path.

## Other code patches
| File | Change |
|------|--------|
| `trellis2/models/__init__.py` | `del state_dict` after load; explicit gc between models |
| `trellis2/pipelines/base.py` | `gc.collect()` + `torch.cuda.empty_cache()` between model loads |
| `trellis2/pipelines/trellis2_image_to_3d.py` | Skip bf16→fp16 conversion (gfx1201 has native bf16) |
| `trellis2/modules/sparse/config.py` | Accept `SPARSE_ATTN_BACKEND=sdpa` |
| `o-voxel/setup.py` | ROCm arch flags (`--offload-arch=gfx1201`), HIP compiler flags |
| `run_amd.py` | `torch.backends.cuda.matmul.allow_{fp16,bf16}_reduced_precision_reduction = False` (fp32 accumulation; numerical safety — did not change the mesh) |

## Notes / known limitations
- `expandable_segments` is unsupported on ROCm (harmless warning); use `PYTORCH_ALLOC_CONF`
  (not the deprecated `PYTORCH_HIP_ALLOC_CONF`).
- o-voxel HIP compilation doesn't build (MSVC include paths with spaces break hipcc); the
  pure-Python hashmap fallback works around it.
- DINOv3 is HF-gated — worked around via the fork's local `MODELS/dinov3` download.
- Occluded-side texture is hallucinated from one view: coherent and plausible after mirroring,
  but not photoreal. A multi-view input or the `1024_cascade` preset would sharpen it.
- Decimation is disabled (`fast_simplification` not installed) → full-res ~90–110 MB GLBs.
  `pip install fast_simplification` to enable quadric decimation.
