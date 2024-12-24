#!/usr/bin/env bash
#
# run.sh
#
# This script activates the supercell_env virtual environment,
# runs the radar_supercell_detection.py script, and then
# drops you into a shell for troubleshooting (still inside the venv).
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

# 4. Drop into a shell for troubleshooting (still in the activated venv)
echo ""
echo "================================================="
echo "The script has finished, but the venv is still active."
echo "Type 'exit' when you are finished troubleshooting."
echo "================================================="
exec $SHELL
