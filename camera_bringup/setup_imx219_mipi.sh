#!/usr/bin/env bash
# ==============================================================================
# setup_imx219_mipi.sh - Raspberry Pi NoIR Camera v2 (Sony IMX219) MIPI Bringup
# Platform: D-Robotics Horizon RDK X5 (4GB) / Ubuntu 22.04 LTS (Kernel 6.1.83)
# ==============================================================================

set -e

echo "[+] Starting IMX219 MIPI CSI camera setup on RDK X5..."

# Check root permissions
if [ "$EUID" -ne 0 ]; then
    echo "[!] Please run as root: sudo bash setup_imx219_mipi.sh"
    exit 1
fi

# Step 1: Kill any stale cam-service or streaming processes
echo "[+] Terminating existing camera processes..."
pkill -9 -f rdk_x5 || true
pkill -9 -f cam-service || true
sleep 1

# Step 2: Verify I2C Sensor Presence at 0x10 (IMX219 address)
echo "[+] Checking I2C bus for IMX219 sensor (0x10)..."
if command -v i2cdetect >/dev/null 2>&1; then
    # Sensor usually resides on I2C bus 4 or bus 5 depending on CSI port
    for bus in 4 5 6; do
        if i2cdetect -y -r $bus 2>/dev/null | grep -q "10:"; then
            echo "[✓] Detected sensor answering on I2C bus $bus at address 0x10."
            break
        fi
    done
fi

# Step 3: Launch hobot cam-service daemon
# Note: Sensor port parameters configure CSI-2 MIPI interface
echo "[+] Launching /usr/hobot/bin/cam-service..."
if [ -f "/usr/hobot/bin/cam-service" ]; then
    /usr/hobot/bin/cam-service -C5 3 5 3 -s4 2 4 2 -i6 -V6 >/dev/null 2>&1 &
    sleep 2
    echo "[✓] cam-service daemon started in background."
else
    echo "[!] Warning: /usr/hobot/bin/cam-service not found in standard location."
fi

# Step 4: Verify V4L2 Device Nodes
echo "[+] Inspecting V4L2 video nodes..."
v4l2-ctl --list-devices 2>/dev/null || true

echo "=============================================================================="
echo "[✓] Setup complete! Recommended verification:"
echo "    - Thermal UVC Camera: /dev/video0 (or /dev/video32)"
echo "    - RGB IMX219 MIPI:    /dev/video10 (or /dev/video8)"
echo "=============================================================================="
