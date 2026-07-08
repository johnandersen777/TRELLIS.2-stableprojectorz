# TRELLIS.2 image-to-3D on AMD Radeon RX 9070 XT (Windows 11, ROCm)

Run the [TRELLIS.2](https://github.com/IgorAherne/TRELLIS.2-stableprojectorz) image→3D
pipeline on an AMD GPU (no NVIDIA/CUDA), producing a clean watertight textured `.glb`.

**Status: working end-to-end.** Verified on three vehicles from a single photo each:
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

- For other images, replace both paths:
  ```bash
  HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
  SPARSE_CONV_BACKEND=none \
  SPARSE_ATTN_BACKEND=sdpa \
  ATTN_BACKEND=sdpa \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -u run_amd.py \
    ../reference-images/Humvee.jpg \
    ../reference-images/Humvee.glb \
    --pipeline-type 512
  ```
- `--no-cache` forces a full regenerate (otherwise a `*.glb.cache.pkl` of the raw mesh is
  reused, so you can iterate on the exporter without re-running inference).
- Runtime: ~2 min after models are cached (1m 39s). First run downloads ~15 GB of checkpoints.

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

---

## TypeScript/Deno Port (WebGPU)

Pure TypeScript port of the TRELLIS.2 pipeline. No Python required for the mesh
pipeline — reads the Python pickle cache and produces identical `.glb` output.
Full WebGPU inference path (SS_Flow + SS_Decoder) is implemented but has a
`toCPU` buffer-validation bug in the Euler sampler loop (under investigation).

**Directory:** `src/` — 47 TypeScript files, zero type errors (`deno check src/`).

### Architecture

```
src/
├── runtime/         GPU context, tensor, shader cache, memory tracker
│   ├── ops/         matmul (tiled WGSL)
│   └── shaders/wgsl/
├── dense/
│   ├── ops/         attention, conv3d, pixel_shuffle_3d, SiLU, GELU,
│   │                elementwise, RMS/LayerNorm, RoPE, linear
│   └── blocks/      timestep_embedder, transformer_cross_block
├── sparse/
│   ├── ops/         attention, conv3d, linear, norm
│   └── blocks/      transformer_cross_block
├── models/          SS_Flow, SS_Decoder, SLatFlow, SparseVAEDecoder
├── samplers/        Flow Euler ODE sampler (CFG + rescale)
├── mesh/            Lewiner MC, GLB writer, symmetry mirroring,
│   │                EDT color fill, gaussian filter, binary closing
│   └── lewiner_tables.ts  (48 decoded lookup tables, 1462 lines)
├── model/           safetensors loader, pipeline.json parser
├── pipeline/        run.ts (CLI), trellis2_image_to_3d.ts
└── validation/      mesh_test, phase0_test, compare harness
```

### VRAM & RAM Requirements

| Mode | Peak VRAM | Peak RAM | Notes |
|------|-----------|----------|-------|
| **Cache mode** (mesh only) | 0 GB | 1.0 GB | CPU-only; reads Python pickle cache |
| **Full inference** (SS_Flow + SS_Decoder) | ~500 MB | ~1.5 GB | Per-block weight streaming, lazy safetensors |
| Python reference (GPU) | 6.1 GB | 23.9 GB | Loads all models + DINOv3 + rembg |

**Why TS is lower:**
- Lazy safetensors reading — per-block weight tensors read from disk on demand
- Per-block GPU disposal — weight buffers freed after each transformer block (28 tensors/block)
- No DINOv3 image feature extractor loaded (Python path only)
- CPU-only mesh pipeline — marching cubes, EDT fill, gaussian filter run on CPU

**Weight sizes (on disk, BF16/FP16 safetensors):**
| Model | File | Size |
|-------|------|------|
| SS_Flow (1.3B params) | `ss_flow_img_dit_1_3B_64_bf16.safetensors` | 2.5 GB |
| SS_Decoder external | `ss_dec_conv3d_16l8_fp16.safetensors` | 141 MB |
| Shape/tex SLatFlow ×4 | Various | 2.5 GB each |
| Shape/tex decoder ×2 | Various | 905 MB each |
| **Total on disk** | | **~14 GB** |

Per-block GPU: SS_Flow loads ~170 MB F32 weights per block (28 tensors), disposes after
block completes. With SS_Decoder (282 MB F32) and intermediate activations (~50-100 MB),
peak VRAM stays under 500 MB with weight streaming.

**Minimum:** 4 GB VRAM, 8 GB RAM.
**Recommended:** 8 GB VRAM, 16 GB RAM.

### How to Run

**Prerequisites:**
- [Deno](https://deno.com/) 2.x+
- AMD Radeon RX 9070 XT (or any WebGPU-capable GPU with 4+ GB VRAM)
- Python 3.12 venv (only for pickle cache extraction; `TRELLIS.2-stableprojectorz/venv/`)
- HuggingFace model cache at `~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/`

**Cache mode** (mesh pipeline only — fast, proven, no GPU needed):
```bash
# Requires pre-computed .glb.cache.pkl from a Python run
deno run --allow-read --allow-write --allow-run --allow-env \
  src/pipeline/run.ts \
  --image reference-images/T-80BVM.jpg \
  --output output.glb
```

This reads `reference-images/T-80BVM.glb.cache.pkl`, extracts coords+attrs via
the venv Python, runs the TS mesh pipeline (occupancy grid → symmetry mirror →
EDT color fill → gaussian → Lewiner MC → GLB), and writes `output.glb`.

**Full inference** (SS_Flow + SS_Decoder — needs WebGPU; see [BUG.md](BUG.md) for mapAsync workaround):
```bash
deno run --unstable-webgpu --allow-env --allow-read --allow-write --allow-ffi --allow-run \
  src/pipeline/run.ts \
  --image reference-images/T-80BVM.jpg \
  --output output.glb \
  --no-cache
```

**Mesh test** (standalone, from pickle cache):
```bash
deno run --allow-read --allow-write --allow-run --allow-env \
  src/validation/mesh_test.ts
# Output: reference-images/T-80BVM-ts.glb
```

**Type check:**
```bash
deno check src/                    # zero errors across 47 files
```

**Serve viewer:**
```bash
deno run --allow-net --allow-read serve.ts &
# Open: http://localhost:8766/glb-viewer.html?model=http://localhost:8766/reference-images/T-80BVM-ts.glb
```

### Mesh Pipeline Output

| Metric | Python GLB | TS GLB |
|--------|-----------|--------|
| Vertices | 2,247,310 | 3,199,790 |
| Faces | 4,500,564 | 4,273,061 |
| Mean color | 0.178, 0.191, 0.163 | 0.178, 0.190, 0.162 |
| Near-gray vertices | 3.7% | 3.2% |
| Mean saturation | 0.222 | 0.223 |
| GLB size | 90 MB | 102 MB |

TS produces more vertices (Lewiner MC vs skimage MC) but near-identical colors
after EDT fill + gaussian σ=1.2. Face count within 5% of Python.

### WebGPU Ops (11 complete)

| Op | Shader | Status |
|----|--------|--------|
| Tiled SDPA attention | WGSL 120 lines | ✅ |
| Dense Conv3d | WGSL implicit GEMM | ✅ |
| Pixel shuffle 3D | WGSL | ✅ |
| SiLU / GELU | WGSL | ✅ |
| Element-wise (add/mul/scale) | WGSL | ✅ |
| RMS / LayerNorm | WGSL | ✅ |
| 3D RoPE | WGSL | ✅ |
| Matmul (tiled) | WGSL | ✅ |
| Linear (W^T transpose) | CPU + matmul | ✅ |

### Known Issues

- **Deno WebGPU `mapAsync` bug** — `GPUBuffer.mapAsync()` fails after heavy
  GPU compute (360+ dispatches). SS_Flow + SS_Decoder compute runs correctly
  on GPU; only the final occupancy→CPU readback fails. No-cache mode falls
  back to Python cache for voxel coords. See [BUG.md](BUG.md) for full
  diagnosis and reproduction steps.
- **Taubin watertightness** — mesh-graph Laplacian smoothing (λ=0.5 / μ=-0.53)
  creates visible gaps. Disabled by default (`taubinIters=0`); opt-in via
  `meshFromVoxels(options)`.
- **No DINOv3 condition** — TS path uses zero conditioning. Python subprocess
  extracts DINOv3 features for the full inference path but is fragile.
- **Model weight path** — expects HuggingFace cache at the default location.
  Override with `--cache-dir` or `TRELLIS_CACHE_DIR`.
