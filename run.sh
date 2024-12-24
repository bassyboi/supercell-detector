#!/usr/bin/env bash
#
# run.sh
#
# This script activates the supercell_env virtual environment and
# runs the BOM radar supercell detection Python script.
#
# Usage:
#   chmod +x run.sh
#   ./run.sh

ENV_DIR="supercell_env"
SCRIPT_NAME="radar_supercell_detection.py"

# 1. Check if the environment exists
if [ ! -d "$ENV_DIR" ]; then
    echo "Error: $ENV_DIR does not exist. Please run ./install.sh first."
    exit 1
fi

# 2. Activate the environment
echo "Activating $ENV_DIR..."
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# 3. Run the Python script
echo "Running $SCRIPT_NAME..."
python "$SCRIPT_NAME"

# 4. (Optional) Deactivate the environment when done
# deactivate
