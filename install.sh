#!/usr/bin/env bash
#
# install.sh
#
# This script creates a Python virtual environment (named `supercell_env`) and
# installs all dependencies needed for the BOM radar supercell detection script.
#
# Usage:
#   1. Make this script executable:
#        chmod +x install.sh
#   2. Run the script (CPU-only):
#        ./install.sh
#      Or to install GPU-enabled PyTorch (example: CUDA 11.8):
#        ./install.sh gpu
#
# After completion, activate the environment with:
#   source supercell_env/bin/activate
#
# Then run:
#   python radar_supercell_detection.py

# -- Settings --
ENV_NAME="supercell_env"
PYTHON_VERSION="python3"  # or "python" depending on your system
GPU_ARG="$1"              # if set to "gpu", will install GPU-enabled torch

# Check if Python is installed
if ! command -v "$PYTHON_VERSION" &> /dev/null
then
    echo "Error: Python not found. Please install Python 3 and re-run."
    exit 1
fi

# 1. Create a virtual environment
echo "Creating Python virtual environment: $ENV_NAME"
"$PYTHON_VERSION" -m venv "$ENV_NAME"

# 2. Activate the environment
echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$ENV_NAME/bin/activate"

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install PyTorch
#    If the user passed "gpu", install a CUDA build.
#    Adjust the CUDA version (cu118, cu117, etc.) to match your system.
if [ "$GPU_ARG" == "gpu" ]; then
  echo "Installing PyTorch (GPU with CUDA 11.8)..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
  echo "Installing PyTorch (CPU-only)..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 5. Install required Python packages
#    - ultralytics: YOLO (v8)
#    - opencv-python: For image processing
#    - pillow, requests, numpy: Common libraries
#    - beautifulsoup4, lxml: For HTML scraping (optional, but useful if scraping BOM pages)
echo "Installing ultralytics (YOLO), OpenCV, Pillow, Requests, NumPy, BeautifulSoup..."
pip install ultralytics opencv-python pillow requests numpy beautifulsoup4 lxml

# 6. Confirm everything is installed
echo "Verifying installations..."
pip list

echo ""
echo "================================================="
echo "Setup complete!"
echo ""
echo "Activate the environment with:"
echo "  source $ENV_NAME/bin/activate"
echo ""
echo "Then run your detection code, for example:"
echo "  python radar_supercell_detection.py"
echo "================================================="

