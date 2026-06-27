# TASK: TRELLIS.2 on AMD Radeon RX 9070 XT (Windows 11)

## Goal
Run TRELLIS.2 image-to-3D pipeline on this AMD GPU. Verify output with `glb-viewer.html` (WebGPU viewer) or `render.ts` (Deno CLI) using reference image `reference-images/T-80BVM.jpg` (T-80 tank). Debug until working, commit each fix.

## Hardware
- **GPU**: AMD Radeon RX 9070 XT (RDNA 4, `gfx1201`) — 15.9 GB VRAM
- **System RAM**: 31 GB
- **OS**: Windows 11 + Git Bash + PowerShell
- **No NVIDIA GPU** — all CUDA paths monkey-patched

## Current State (2026-06-27 00:15)

### ✅ Working — Pipeline Runs End-to-End
- **ROCm PyTorch 2.9.1** installed in `TRELLIS.2-stableprojectorz/venv/`
- **GPU detected**: `torch.cuda.is_available() == True`, "AMD Radeon RX 9070 XT"
- **All models downloaded and cached**: TRELLIS.2-4B (7 checkpoints, ~14.8GB), TRELLIS-image-large (ss_dec), DINOv3 (MODELS/dinov3), RMBG-2.0 (MODELS/RMBG-2.0)
- **Pipeline imports and loads all models** (~19GB peak RAM during load, stabilizes at ~15GB)
- **SDPA sparse attention works** — `SPARSE_ATTN_BACKEND=sdpa` confirmed working (all 3 sampling phases complete in ~6s total GPU time)
- **All sampling phases complete** — sparse structure (1.0s), shape SLat (4s), texture SLat (0.4s)
- **Mesh generated**: 689,639 verts, 1,021,852 faces
- **GLB exported**: `reference-images/T-80BVM.glb` (18MB, 529K verts after cleaning, 1M faces)
- **Verified in glb-viewer.html** via Chrome — loads and renders (1 draw call)
- **Pickle cache** implemented in run_amd.py — `--no-cache` to regenerate

### ❌ Quality Issue: GLB output is messy/junk
The generated mesh looks wrong in the viewer — not a clean T-80 tank. Root cause identified:

**Dict-based hashmap fallback used wrong hash function.** The CUDA kernel uses `flat_idx = x*gy*gz + y*gz + z` while the dict fallback used `(x*p1 ^ y*p2 ^ z*p3) % capacity`. Different hash space → wrong vertex correspondences from dual contouring → garbled mesh.

**Fix applied but untested**: Rewrote `hashmap_fallback.py` to use GPU-sorted `flat_idx` arrays with `torch.searchsorted` (matches CUDA flat-index approach). Expected to fix quality AND speed (~10ms GPU vs ~5 min CPU).

### 🔧 To fix quality: Re-run with GPU-sorted hashmap
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
Expect: ~2 min total (GPU-sorted hashmap removes CPU bottleneck). Output should be clean T-80 tank mesh.

## Fallback Modules Created

### `trellis2/modules/sparse/conv/conv_none.py`
Pure-PyTorch sparse submanifold conv. Weight layout matches flex_gemm checkpoint: (Co, Kd, Kh, Kw, Ci). Uses linear-probe neighbor map. `SPARSE_CONV_BACKEND=none`.

### `hashmap_fallback.py`
GPU-sorted flat_idx hashmap (v3). Replaces o-voxel._C hashmap functions. Uses `torch.sort` + `torch.searchsorted` for O(log N) binary search on GPU. Zero Python loops, zero D2H transfers. Matches CUDA flat_idx formula: `x*gy*gz + y*gz + z`.

### `export_glb_fallback.py`
trimesh-based GLB export. No cumesh/nvdiffrast needed. Mesh cleaning + quadric decimation. Handles trimesh 4.x API changes gracefully.

### `run_amd.py`
Main AMD runner. Registers dummy modules (flex_gemm, nvdiffrast, cumesh with CuMesh dummy, flash_attn, xformers). Loads o-voxel Python sources with fallback hashmap _C. Pickle mesh cache (`--no-cache` to skip). Sets environment, runs pipeline.

### `setup.sh`
AMD ROCm setup script. Install Python 3.12, ROCm SDK, PyTorch 2.9.1, dependencies.

## All Code Patches (need committing)

| File | Change |
|------|--------|
| `trellis2/models/__init__.py` | Added `del state_dict` after load; explicit gc between models |
| `trellis2/pipelines/base.py` | Added `gc.collect()` + `torch.cuda.empty_cache()` between model loads |
| `trellis2/pipelines/trellis2_image_to_3d.py` | Skip bf16→fp16 conversion (AMD gfx1201 supports native bf16) |
| `trellis2/modules/sparse/config.py` | Added `'sdpa'` to accepted attention backends |
| `trellis2/modules/sparse/attention/full_attn.py` | Added `sdpa` branch for variable-length SDPA |
| `o-voxel/setup.py` | ROCm arch flags (`--offload-arch=gfx1201`), HIP compiler flags |
| `trellis2/modules/sparse/conv/conv_none.py` | **New file** — pure-PyTorch sparse conv |
| `hashmap_fallback.py` | **New file** — GPU-sorted flat_idx hashmap (v3: sort+searchsorted) |
| `export_glb_fallback.py` | **New file** — trimesh GLB export (trimesh 4.x compat) |
| `run_amd.py` | **New file** — AMD runner with CuMesh dummy + pickle cache |
| `setup.sh` | **New file** — AMD ROCm setup script |

## Known Issues
- **expandable_segments not supported on ROCm** — warning, harmless
- **`PYTORCH_HIP_ALLOC_CONF` deprecated** — use `PYTORCH_ALLOC_CONF` instead
- **o-voxel HIP compilation not working** — MSVC include paths with spaces break hipcc. Fallback hashmap works around this.
- **Model loading peaks at ~19GB RAM** — acceptable for 31GB system.
- **DINOv3 requires HF gated access** — worked around via fork's local MODELS/dinov3 download from GitHub releases.
- **⚡ GLB mesh quality is wrong** — dict hashmap used wrong hash function (xorshift vs flat_idx). GPU-sorted flat_idx hashmap fix applied, needs re-run to verify.
- **Texture/PBR attributes may be missing or wrong** — export_glb_fallback.py uses attrs volume for PBR channels. Needs verification after hash function fix.
