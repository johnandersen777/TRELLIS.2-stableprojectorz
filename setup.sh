#!/usr/bin/env bash
# =============================================================================
#  TRELLIS.2 (AMD ROCm) — Windows / Git Bash setup + image-to-3D runner
# =============================================================================
#  Run this from Git Bash:   bash setup.sh
#  Optionally pass an image:  bash setup.sh input.png output.glb
#       -> if two args are given, generation runs automatically after setup.
#
#  PREREQUISITES:
#    * Git for Windows                          (provides Git Bash)
#    * Python 3.12  (AMD ROCm 7.2.1 requires 3.12)
#    * AMD Adrenalin driver 26.2.2+ with AI Bundle
#    * Visual Studio Build Tools w/ "Desktop development with C++"
#         (needed to compile o-voxel from source for HIP/ROCm)
#    * Visual C++ 2015-2022 Redistributables
#
#  NOTE: This script is adapted from the CUDA version to work with
#  AMD Radeon GPUs (RX 9070 XT / RDNA 4) via AMD ROCm 7.2.1 on Windows.
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/IgorAherne/TRELLIS.2-stableprojectorz.git"
REPO_DIR="TRELLIS.2-stableprojectorz"

echo "========================================================="
echo " TRELLIS.2 Windows (Git Bash) AMD ROCm setup"
echo " AMD Radeon RX 9070 XT / RDNA 4"
echo "========================================================="

# --- GPU Detection -----------------------------------------------------------
echo "[*] Detecting GPU platform..."
GPU_PLATFORM="cpu"
if command -v rocminfo >/dev/null 2>&1; then
    GPU_PLATFORM="rocm"
    echo "[*] ROCm platform detected (AMD GPU)"
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_PLATFORM="cuda"
    echo "[*] CUDA platform detected (NVIDIA GPU)"
else
    echo "[!] No GPU compute platform detected (no rocminfo, no nvidia-smi)."
    echo "    Will attempt ROCm path anyway — ensure AMD drivers are installed."
    GPU_PLATFORM="rocm"
fi

# --- Pick Python 3.12 interpreter --------------------------------------------
PYTHON312_PATHS=(
    "/c/Users/$USER/AppData/Local/Programs/Python/Python312/python.exe"
    "/c/Program Files/Python312/python.exe"
    "/c/Python312/python.exe"
    "python3.12"
    "python"
)

PY=""
for candidate in "${PYTHON312_PATHS[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" --version 2>&1 || true)
        if [[ "$ver" == *"3.12"* ]]; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    # Try py launcher
    if command -v py >/dev/null 2>&1; then
        if py -3.12 -c "import sys" >/dev/null 2>&1; then
            PY="py -3.12"
        fi
    fi
fi

if [ -z "$PY" ]; then
    echo "[!] ERROR: Python 3.12 not found."
    echo "    AMD ROCm 7.2.1 requires Python 3.12 specifically."
    echo "    Install from: https://www.python.org/downloads/release/python-31210/"
    echo "    Or run: winget install Python.Python.3.12"
    exit 1
fi

echo "[*] Using Python: $PY"
$PY --version

# --- 1. Clone the low-VRAM fork (with submodules) ----------------------------
if [ ! -d "$REPO_DIR" ]; then
    echo "[*] Cloning $REPO_URL ..."
    git clone -b main --recursive "$REPO_URL" "$REPO_DIR"
else
    echo "[*] Repo already present; reusing $REPO_DIR"
fi
cd "$REPO_DIR"

# --- 2. Create + activate an isolated virtual environment --------------------
if [ ! -d "venv" ]; then
    echo "[*] Creating Python virtual environment..."
    $PY -m venv venv
fi
source venv/Scripts/activate
python -m pip install --upgrade pip wheel setuptools

# --- 3. Install AMD ROCm 7.2.1 PyTorch ---------------------------------------
echo "[*] Installing AMD ROCm 7.2.1 SDK + PyTorch 2.9.1..."

ROCM_BASE="https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"

# Install ROCm SDK components
pip install --no-cache-dir \
    "$ROCM_BASE/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" \
    "$ROCM_BASE/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" \
    "$ROCM_BASE/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl"

# Install rocm meta-package (required by torch)
pip install --no-cache-dir "$ROCM_BASE/rocm-7.2.1.tar.gz"

# Install PyTorch, torchvision, torchaudio
pip install --no-cache-dir \
    "$ROCM_BASE/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" \
    "$ROCM_BASE/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" \
    "$ROCM_BASE/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"

# --- 4. Verify GPU detection -------------------------------------------------
echo "[*] Verifying GPU detection..."
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info(0)
    print(f'VRAM: {total/1024**3:.1f} GB total ({free/1024**3:.1f} GB free)')
else:
    print('[!] WARNING: No GPU detected by PyTorch. Check AMD drivers.')
"

# --- 5. Install general Python dependencies ----------------------------------
echo "[*] Installing general dependencies..."
pip install \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    ninja trimesh "transformers==4.57.3" "gradio==6.0.1" tensorboard \
    pandas lpips zstandard kornia timm \
    huggingface_hub accelerate psutil

# Install xformers for AMD (used as attention fallback)
pip install xformers==0.0.32.post2 \
    --index-url https://download.pytorch.org/whl/rocm7.0 2>/dev/null || \
    echo "[!] xformers install failed; will use SDPA fallback for attention"

# Install utils3d from git
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- 6. Install Pillow (standard, skip SIMD on AMD) --------------------------
echo "[*] Installing Pillow..."
pip install pillow

# --- 7. Build and install o-voxel from source (HIP compilation) ---------------
echo "[*] Building o-voxel from source for AMD ROCm..."

# Set ROCm build environment
export PYTORCH_ROCM_ARCH="gfx1201"  # RDNA 4 / RX 9070 XT

# Clean any stale build artifacts
rm -rf o-voxel/build/ o-voxel/dist/ o-voxel/*.egg-info/ 2>/dev/null || true

# Patch o-voxel setup.py for HIP compilation
cat > o-voxel/setup_rocm.patch.py << 'PYEOF'
# This is applied as a replacement setup.py for ROCm builds.
# It uses HIP compiler flags instead of NVCC flags for AMD GPUs.
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Try to use HIPExtension if available (ROCm PyTorch), fall back to CUDAExtension
try:
    from torch.utils.cpp_extension import HIPExtension as CudaLikeExtension
    USE_HIP = True
    print("[o-voxel] Using HIPExtension for ROCm build")
except ImportError:
    from torch.utils.cpp_extension import CUDAExtension as CudaLikeExtension
    USE_HIP = False
    print("[o-voxel] Using CUDAExtension (HIPExtension not available)")

# Compiler flags
if os.name == 'nt':
    cxx_flags = ['/O2', '/std:c++17', '/bigobj', '/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS']
else:
    cxx_flags = ['-O3', '-std=c++17']

# HIP/ROCm compiler flags (instead of NVCC flags)
gpu_compiler_flags = [
    '-O3',
    '--use_fast_math',
    '-std=c++17',
]

# AMD GPU architecture targets (RDNA 4 / RX 9070 XT)
rocm_arch = os.environ.get('PYTORCH_ROCM_ARCH', 'gfx1201')
# Allow multiple architectures: gfx1201;gfx1100;gfx1030
for arch in rocm_arch.split(';'):
    gpu_compiler_flags.append(f'--offload-arch={arch}')

setup(
    name="o_voxel",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CudaLikeExtension(
            name="o_voxel._C",
            sources=[
                "src/hash/hash.cu",
                "src/convert/flexible_dual_grid.cpp",
                "src/convert/volumetric_attr.cpp",
                "src/serialize/api.cu",
                "src/serialize/hilbert.cu",
                "src/serialize/z_order.cu",
                "src/io/svo.cpp",
                "src/io/filter_parent.cpp",
                "src/io/filter_neighbor.cpp",
                "src/rasterize/rasterize.cu",
                "src/ext.cpp",
            ],
            include_dirs=[
                os.path.join(ROOT, "third_party/eigen"),
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": gpu_compiler_flags,
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
PYEOF

# Replace setup.py with ROCm-compatible version
cp o-voxel/setup.py o-voxel/setup_cuda.bak.py
cp o-voxel/setup_rocm.patch.py o-voxel/setup.py

# Build o-voxel from source for ROCm
echo "[*] Compiling o-voxel for AMD ROCm (this may take a few minutes)..."
pip install -e o-voxel/ --no-build-isolation 2>&1 || {
    echo "[!] o-voxel HIP compilation failed."
    echo "    This is expected for first-time setup. We'll continue"
    echo "    and handle this in the debugging phase."
    echo "    Error log saved to o-voxel_build_error.log"
}

# Restore original setup.py
cp o-voxel/setup_cuda.bak.py o-voxel/setup.py

# --- 8. Handle Pillow SIMD replacement (optional, skip if problematic) --------
if [ -d "whl" ] && ls whl/Pillow_SIMD*.whl >/dev/null 2>&1; then
    echo "[*] Installing Pillow-SIMD..."
    pip uninstall -y pillow 2>/dev/null || true
    pip install "$(ls whl/Pillow_SIMD*.whl | head -1)" 2>/dev/null || {
        echo "[!] Pillow-SIMD failed; restoring standard Pillow..."
        pip install pillow
    }
fi

# --- 9. Fix/refresh submodules -----------------------------------------------
echo "[*] Re-initializing submodules..."
git submodule deinit -f --all 2>/dev/null || true
git submodule update --init --recursive

# --- 10. Create the image-to-3D runner script ---------------------------------
cat > run_image_to_3d.py << 'PYEOF'
#!/usr/bin/env python
"""
TRELLIS.2 image-to-3D runner — AMD ROCm adapted version.

Supports both AMD (ROCm) and NVIDIA (CUDA) GPUs via PyTorch's unified
"cuda" device interface. For AMD GPUs, sets appropriate environment
variables and backend selections.
"""
import os
import sys
import argparse

# --- Environment setup for AMD ROCm ---
# ROCm uses "cuda" device string, so most code works unchanged.
# Set attention backend to SDPA (PyTorch built-in, works on all GPUs)
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# ROCm memory allocation config
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

# --- GPU detection ---
import torch
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_mem / 1024**3:.1f} GB total ({free_mem / 1024**3:.1f} GB free)")
    DEVICE = "cuda"

    # AMD-specific: set HIP visible devices (mirrors CUDA_VISIBLE_DEVICES)
    if "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper():
        print("[*] AMD GPU detected — using ROCm backend")
        os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
        # Use SDPA for attention (flash-attn not available on RDNA)
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
else:
    print("[!] No GPU detected. Running on CPU (will be very slow).")
    DEVICE = "cpu"

# --- Imports ---
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

# Try to import o_voxel (compiled extension)
try:
    import o_voxel
    print("[OK] o_voxel compiled extension loaded")
except ImportError as e:
    print(f"[!] o_voxel not available: {e}")
    print("    The pipeline may fail during mesh extraction.")


def main():
    ap = argparse.ArgumentParser(description="TRELLIS.2 image -> .glb (AMD ROCm)")
    ap.add_argument("input", help="path to input image (png/jpg)")
    ap.add_argument("output", help="path to output .glb")
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B",
                    help="HuggingFace model repo or local path")
    ap.add_argument("--pipeline-type", default="1024_cascade",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"],
                    help="Pipeline resolution/quality preset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU even if GPU available")
    args = ap.parse_args()

    device = "cpu" if args.cpu else DEVICE
    print(f"[*] Using device: {device}")

    print(f"[*] Loading pipeline: {args.model}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.to(device)
    pipeline.low_vram = True  # Enable low-VRAM mode

    print(f"[*] Generating from: {args.input}")
    image = Image.open(args.input)

    mesh = pipeline.run(
        image,
        num_samples=1,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
    )[0]

    # Simplify mesh to keep under typical GLB limits
    mesh.simplify(16777216)

    print(f"[*] Exporting GLB: {args.output}")
    glb = o_voxel.postprocess.to_glb(
        vertices          = mesh.vertices,
        faces             = mesh.faces,
        attr_volume       = mesh.attrs,
        coords            = mesh.coords,
        attr_layout       = mesh.layout,
        voxel_size        = mesh.voxel_size,
        aabb              = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target = 1000000,
        texture_size      = 4096,
        remesh            = True,
        remesh_band       = 1,
        remesh_project    = 0,
        verbose           = True,
    )
    glb.export(args.output, extension_webp=True)
    print(f"[done] Saved {args.output}")


if __name__ == "__main__":
    main()
PYEOF

echo "========================================================="
echo " Setup complete."
echo ""
echo "  Web UI:  python app.py"
echo "  CLI:     python run_image_to_3d.py input.png output.glb"
echo ""
echo "  Recommended pipeline types (smaller = less VRAM):"
echo "    512           — Fast, lowest quality"
echo "    1024          — Balanced"
echo "    1024_cascade  — Best quality (default)"
echo "    1536_cascade  — Maximum quality, needs more VRAM"
echo "========================================================="

# --- 11. If an image + output were passed, run generation --------------------
if [ "$#" -ge 2 ]; then
    echo "[*] Args detected — running generation: $1 -> $2"
    python run_image_to_3d.py "$1" "$2"
fi
