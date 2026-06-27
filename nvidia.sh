#!/usr/bin/env bash
# =============================================================================
#  TRELLIS.2 (low-VRAM) — Windows / Git Bash setup + image-to-3D runner
# =============================================================================
#  Run this from Git Bash (NOT cmd):   bash setup_trellis2.sh
#  Optionally pass an image + output:  bash setup_trellis2.sh input.png output.glb
#       -> if two args are given, generation runs automatically after setup.
#
#  PREREQUISITES (install + add to PATH first):
#    * Git for Windows                          (provides Git Bash)
#    * Python 3.11  (check "Add python to PATH")
#    * NVIDIA driver supporting CUDA 12.8
#    * Visual Studio Build Tools w/ "Desktop development with C++"
#         -> only needed if a prebuilt wheel is missing and pip must compile.
#
#  EASIEST ALTERNATIVE: the maintainer ships a one-click Windows installer
#  (Python 3.11 / CUDA 12.8 / Torch 2.8). If this script gives you trouble, use:
#    https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/tag/latest
#
#  NOTE ON VRAM: this fork is optimized for ~8GB GPUs. The reference low-VRAM
#  path is the bundled app.py (web UI). The CLI wrapper written below uses the
#  documented pipeline API + expandable_segments allocator; on very tight VRAM
#  prefer:  python app.py
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/IgorAherne/TRELLIS.2-stableprojectorz.git"
REPO_DIR="TRELLIS.2-stableprojectorz"
# Official upstream (Linux / 24GB) if you ever want it instead:
#   https://github.com/microsoft/TRELLIS.2.git

echo "========================================================="
echo " TRELLIS.2 Windows (Git Bash) low-VRAM setup"
echo "========================================================="

# --- Pick a Python 3.11 interpreter ------------------------------------------
if command -v py >/dev/null 2>&1 && py -3.11 -c "import sys" >/dev/null 2>&1; then
    PY="py -3.11"
elif command -v python >/dev/null 2>&1; then
    PY="python"
    echo "[!] 'py -3.11' not found; falling back to 'python'."
    echo "    Make sure it is Python 3.11 — newer/older versions break the prebuilt wheels."
else
    echo "[!] ERROR: No Python found on PATH. Install Python 3.11 first."
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
echo "[*] Creating Python virtual environment..."
$PY -m venv venv
# Windows venvs keep their activate script under Scripts/ (works in Git Bash):
source venv/Scripts/activate
python -m pip install --upgrade pip wheel setuptools

# --- 3. Install PyTorch matching this fork (CUDA 12.8 / Torch 2.8) -----------
echo "[*] Installing PyTorch (CUDA 12.8 build)..."
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# --- 4. Install the PREBUILT CUDA extension wheels (o-voxel, flexgemm, etc.) --
# These ship in the repo's whl/ folder specifically so Windows users don't have
# to compile them. Installing them is what makes this work without a full
# CUDA-from-source build.
if [ -d "whl" ] && ls whl/*.whl >/dev/null 2>&1; then
    echo "[*] Installing prebuilt wheels from whl/ ..."
    pip install whl/*.whl
else
    echo "[!] No wheels found in whl/. They are required on Windows."
    echo "    Grab the one-click release instead:"
    echo "    https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/tag/latest"
fi

# --- 5. Install the remaining Python dependencies ----------------------------
# This fork drives dependency installation through install.py
# (see install_dependencies() around line ~130). Run its installer:
if [ -f "install.py" ]; then
    echo "[*] Running repo dependency installer (install.py)..."
    echo "    If it stalls or expects conda, open install.py and run the pip"
    echo "    commands inside install_dependencies() manually."
    python install.py || {
        echo "[!] install.py did not complete cleanly — check its output above."
    }
else
    echo "[!] install.py not found; verify the repo cloned correctly."
fi

# --- 6. Fix/refresh submodules (explicit step the maintainer documents) ------
echo "[*] Re-initializing submodules..."
git submodule deinit -f --all || true
git submodule update --init --recursive

# --- 7. Write a small command-line image->GLB runner -------------------------
# Built from the repo's documented pipeline API. Weights (microsoft/TRELLIS.2-4B,
# ~several GB) download automatically from Hugging Face on first run.
cat > run_image_to_3d.py <<'PYEOF'
import os, sys, argparse
# Memory + EXR flags (documented in the TRELLIS.2 example):
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: F401  (ensures CUDA context is set up)
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


def main():
    ap = argparse.ArgumentParser(description="TRELLIS.2 image -> .glb")
    ap.add_argument("input", help="path to input image (png/jpg)")
    ap.add_argument("output", help="path to output .glb")
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    args = ap.parse_args()

    print(f"[*] Loading pipeline: {args.model}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    print(f"[*] Generating from: {args.input}")
    image = Image.open(args.input)          # pipeline handles preprocessing
    mesh = pipeline.run(image)[0]
    mesh.simplify(16777216)                 # nvdiffrast vertex limit

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
echo " Setup finished."
echo "   Web UI (best for 8GB VRAM):   python app.py"
echo "   CLI:   python run_image_to_3d.py input.png output.glb"
echo "========================================================="

# --- 8. If an image + output were passed in, run generation now --------------
if [ "$#" -ge 2 ]; then
    echo "[*] Args detected — running generation: $1 -> $2"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python run_image_to_3d.py "$1" "$2"
fi
