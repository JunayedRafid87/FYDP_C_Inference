# 01. Hardware & Dual-Camera Bringup Guide

This document covers the complete hardware configuration and kernel driver setup for running dual-modality vision (RGB + Thermal) on the **D-Robotics Horizon RDK X5 (4GB)** edge computing platform running Ubuntu 22.04 LTS (Linux Kernel 6.1.83).

---

## 1. Hardware Bill of Materials (BOM)

| Component | Specification | Interface / Bus | Kernel Node |
|---|---|---|---|
| **Edge Compute Board** | D-Robotics Horizon RDK X5 (4GB LPDDR4, 8-Core ARM Cortex-A55, Dual BPU Bayes-e 10 TOPS) | System on Module | `/dev/mem`, `/dev/ion` |
| **RGB Camera** | Raspberry Pi NoIR Camera Module v2 (Sony IMX219, 8MP, 1080p/720p) | 22-pin FFC MIPI CSI-2 | `/dev/video10` (or `/dev/video8`) |
| **Thermal Camera** | Senxor / InfiRay Micro-Core UVC Module (256×192 LWIR, 25 Hz) | USB 2.0 / UVC | `/dev/video0` (or `/dev/video32`) |
| **Gas Telemetry** | MQ-series gas sensor / Arduino / STM32 bridge | UART / USB-Serial | `/dev/ttyUSB0` or simulated daemon |
| **Tether / Uplink** | 30 m Cat6 Gigabit Ethernet / 5 GHz Wi-Fi 802.11ac | RJ45 / wlan0 | `eth0` / `wlan0` |

---

## 2. IMX219 MIPI CSI-2 Interface Configuration

The Raspberry Pi NoIR Camera v2 utilizes a Sony IMX219 sensor communicating over MIPI CSI-2 with I2C control at address `0x10`.

### Physical Connection
- Connect the 22-pin FFC ribbon to the CSI-2 connector (closer to the SoC / power input).
- Ensure the blue backing faces the Ethernet port and silver pins contact the PCB side.

### I2C Sensor Presence Verification
Verify that the sensor acknowledges I2C polling:
```bash
sudo i2cdetect -y -r 4
# Expected output: 0x10 responds with "10" or "UU" (kernel driver bound)
```

### Launching `cam-service`
The Horizon BSP relies on `cam-service` to configure the camera subsystem, sensor clocking, and MIPI Virtual Channel routing:
```bash
sudo pkill -9 -f cam-service
sudo /usr/hobot/bin/cam-service -C5 3 5 3 -s4 2 4 2 -i6 -V6 &
```

*Parameter Breakdown:*
- `-C5 3 5 3`: Configures CSI-2 lane mapping and sensor profile.
- `-s4 2 4 2`: Configures sensor interface clocks.
- `-i6 -V6`: Routes the sensor pipeline to `/dev/video6` / `/dev/video10`.

---

## 3. UVC Thermal Camera Bringup

The thermal camera enumerates as a standard USB Video Class (UVC) device outputting raw 16-bit radiometric frames (Y16) or 8-bit grayscale at 256×192 resolution:

```bash
# Query V4L2 device parameters
v4l2-ctl -d /dev/video0 --get-fmt-video
# Verify frame capture
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=10
```

---

## 4. Common Failure Modes & Troubleshooting

### Issue 1: Solid Green Frames in NV12
- **Cause**: NV12 chroma (UV) plane contains all zeros (`0x00`), which decodes as peak green in YUV-to-RGB space.
- **Fix**: The capture buffer was not populated by the ISP before dequeueing. Ensure `cam-service` is running and the V4L2 pixel format is set to `V4L2_PIX_FMT_NV12` or `V4L2_PIX_FMT_BGR24`.

### Issue 2: `VIDIOC_REQBUFS` / Device Busy
- **Cause**: Another process (e.g., previous streamer instance) is holding the `/dev/video*` file descriptor.
- **Fix**: Run `sudo pkill -9 -f rdk_x5` before launching a new streaming session.

### Issue 3: Frame Rate Drops (<3 FPS) under MIPI Load
- **Cause**: MIPI link desynchronization or buffer queue overflow in kernel driver.
- **Fix**: Reboot the board (`sudo reboot`) or re-initialize `cam-service` using `setup_imx219_mipi.sh`.
