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
#   2. Run the script:
#        ./install.sh
#
# After completion, activate the environment with:
#   source supercell_env/bin/activate
#
# Then you can run the supercell detection code in that environment.

# -- Settings --
ENV_NAME="supercell_env"
PYTHON_VERSION="python3"  # or "python", depending on your system

# Check if Python is installed
if ! command -v $PYTHON_VERSION &> /dev/null
then
    echo "Error: Python not found. Please install Python 3 and re-run."
    exit 1
fi

# 1. Create a virtual environment
echo "Creating Python virtual environment: $ENV_NAME"
$PYTHON_VERSION -m venv $ENV_NAME

# 2. Activate the environment
echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$ENV_NAME/bin/activate"

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install PyTorch (CPU or GPU)
#    Adjust this line according to your system’s CUDA version, if you have a GPU.
#    For CPU-only:
echo "Installing PyTorch (CPU-only)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Install required Python packages
echo "Installing ultralytics (for YOLO), OpenCV, Pillow, Requests, NumPy..."
pip install ultralytics opencv-python pillow requests numpy

# 6. (Optional) If you want to use the YOLOv5 repo directly:
# pip install git+https://github.com/ultralytics/yolov5.git@master

# 7. Confirm everything is installed
echo "Verifying installations..."
pip list

echo ""
echo "================================================="
echo "Setup complete!"
echo "To start using this environment, run:"
echo "  source $ENV_NAME/bin/activate"
echo "Then run:"
echo "  python radar_supercell_detection.py"
echo "================================================="
