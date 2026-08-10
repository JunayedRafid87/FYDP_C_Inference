#!/usr/bin/env bash
# ==============================================================================
# convert_yolo11_to_bpu.sh - End-to-End YOLO11 to D-Robotics BPU Conversion Script
# Requires: Docker + openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 image
# ==============================================================================

set -e

TOOLCHAIN_IMAGE="openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8"
MODEL_CONFIG=${1:-"yamls/thermal_yolo11n_v3_bayese_640x640_nv12.yaml"}

echo "=============================================================================="
echo "[+] Starting BPU Model Compilation with OpenExplorer Toolchain"
echo "    Config: $MODEL_CONFIG"
echo "    Image:  $TOOLCHAIN_IMAGE"
echo "=============================================================================="

# Check if Docker is available
if ! command -v docker >/dev/null 2>&1; then
    echo "[-] Error: docker is not installed or not in PATH."
    exit 1
fi

# Prepare directories
mkdir -p output cal

# Verify YAML config exists
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "[-] Error: Config file $MODEL_CONFIG not found."
    exit 1
fi

# Launch Docker conversion container
echo "[+] Executing hb_mapper makertbin inside OpenExplorer container..."
docker run --rm -v "$(pwd):/workspace" -w /workspace "$TOOLCHAIN_IMAGE" /bin/bash -c "
    echo '[Docker] Checking hb_mapper version...'
    hb_mapper --version || true
    echo '[Docker] Running compilation...'
    hb_mapper makertbin --config $MODEL_CONFIG --model-type onnx
"

echo "[✓] Compilation completed! Check output/ directory for compiled .bin model files."
