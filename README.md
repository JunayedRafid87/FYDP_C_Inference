# FYDP_C_Inference: Dual-Modality Edge AI Inference & Teleoperation Pipeline

[![Platform](https://img.shields.io/badge/Platform-D--Robotics%20Horizon%20RDK%20X5-blue.svg)](https://developer.d-robotics.cc/)
[![BPU](https://img.shields.io/badge/BPU-Bayes--e%2010%20TOPS-green.svg)](https://developer.d-robotics.cc/)
[![Model](https://img.shields.io/badge/Model-YOLO11%20Quantized%20INT8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![C/C++](https://img.shields.io/badge/Engine-Native%20C%2FC%2B%2B%20Accelerated-purple.svg)](./c_inference)
[![Release](https://img.shields.io/badge/Version-v10%20(Latest%20Production)-brightgreen.svg)](./pipeline/rdk_x5_stream_ground_v10.py)

Comprehensive Final Year Design Project (FYDP) edge computing repository implementing high-speed dual-modality computer vision (IMX219 NoIR CSI RGB + Senxor 256×192 UVC Thermal), hardware-accelerated BPU inference, real-time telemetry, and a responsive Ground Station web console on the **D-Robotics Horizon RDK X5 (4GB)** edge computing board.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Hardware Bill of Materials](#hardware-bill-of-materials)
3. [Repository Directory Structure](#repository-directory-structure)
4. [Pipeline Evolution & Version History (v1 - v10)](#pipeline-evolution--version-history-v1---v10)
5. [Performance Benchmarks & Profiling](#performance-benchmarks--profiling)
6. [Native C/C++ Acceleration Engine](#native-cc-acceleration-engine)
7. [Quickstart & Usage Guide](#quickstart--usage-guide)
8. [Technical Documentation Index](#technical-documentation-index)

---

## System Architecture

```
+-------------------------------------------------------------------------------------------------+
|                                 ROVER HARDWARE PLATFORM (RDK X5)                                |
|                                                                                                 |
|   +--------------------------+    +--------------------------+    +--------------------------+  |
|   | Raspberry Pi NoIR v2     |    | Senxor UVC LWIR Module   |    | MQ-Series Gas Sensor     |  |
|   | (Sony IMX219 8MP CSI-2)  |    | (256x192 Thermal /dev/v0)|    | (UART Telemetry)         |  |
|   +------------+-------------+    +------------+-------------+    +------------+-------------+  |
|                |                               |                               |                |
|                v                               v                               v                |
|   +------------------------------------------------------------------------------------------+  |
|   |                         Linux Kernel V4L2 Subsystem & cam-service                        |  |
|   +--------------------------------------------+---------------------------------------------+  |
|                                                | Zero-Copy NV12 Tensors                         |
|                                                v                                                |
|   +------------------------------------------------------------------------------------------+  |
|   |                    D-Robotics Horizon BPU Acceleration Core (Bayes-e)                    |  |
|   |    - RGB Model:     yolo11m_detect_bayese_640x640_nv12.bin   (~5.2 ms forward)           |  |
|   |    - Thermal Model: thermal_yolo11n_v3_bayese_640x640_nv12.bin (~5.2 ms forward)        |  |
|   +--------------------------------------------+---------------------------------------------+  |
|                                                | Multi-Scale Output Tensors (6 Feature Maps)    |
|                                                v                                                |
|   +------------------------------------------------------------------------------------------+  |
|   |            High-Performance C Engine / Python Pipeline (rdk_x5_stream_ground_v10)         |  |
|   |    - Vectorized DFL Box Decoding & Sigmoid LUT                                           |  |
|   |    - Class-Specific Non-Maximum Suppression (NMS)                                        |  |
|   |    - White-Hot Radiometric Colormapping & Multi-Threaded MJPEG Broadcaster               |  |
|   +--------------------------------------------+---------------------------------------------+  |
|                                                | Port 8080 (MJPEG Streams + JSON Telemetry)      |
+------------------------------------------------+------------------------------------------------+
                                                 | (Wi-Fi 5 GHz / 30m Cat6 Ethernet Tether)
                                                 v
+-------------------------------------------------------------------------------------------------+
|                       OPERATOR GROUND STATION CONSOLE (Rover_GS_fix_v1.html)                    |
|   - Dual Real-Time Feeds (RGB 30 FPS, Thermal 15 FPS with image-rendering: crisp-edges)        |
|   - AI Bounding Box Canvas Overlay (Dynamic Client-Side Rendering)                              |
|   - System Telemetry HUD (SOM Temps, DDR/BPU/CPU Clocks, RAM Usage, Gas Level Graphs)          |
+-------------------------------------------------------------------------------------------------+
```

---

## Hardware Bill of Materials

| Component | Technical Specifications | Purpose in Rover System |
|---|---|---|
| **SBC** | D-Robotics Horizon RDK X5 (4GB LPDDR4, 8-Core ARM Cortex-A55, 10 TOPS BPU) | Central on-board processing unit |
| **RGB Sensor** | Raspberry Pi NoIR Camera v2 (Sony IMX219, 8MP, 720p/1080p) | Primary teleoperation & obstacle detection |
| **Thermal Sensor** | Senxor / InfiRay Micro-Core UVC (256×192 LWIR, 25 Hz) | Night vision, heat signature, & victim localization |
| **Gas Sensor** | MQ-Series / Analog Gas Sensor Module with UART ADC bridge | Hazardous atmospheric gas detection |
| **Connectivity** | 30 m Cat6 Gigabit Ethernet / 5 GHz 802.11ac Wi-Fi | High-bandwidth, low-latency command link |

---

## Repository Directory Structure

```
FYDP_C_Inference/
├── README.md                                # Master project documentation
├── LICENSE                                  # MIT Open-Source License
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore configuration
│
├── pipeline/                                # Streaming & Inference Pipeline Versions (v1 - v10)
│   ├── rdk_x5_stream_ground_v10.py          # [LATEST STABLE PRODUCTION] Dual BPU + Live Telemetry
│   ├── rdk_x5_stream_ground_v9.py           # v9: Clean rollback + cosmetic CSS fixes
│   ├── rdk_x5_stream_ground_v8.py           # v8: Inlined web UI attempt
│   ├── rdk_x5_stream_ground_v7.py           # v7: Grayscale thermal palette + crisp-edges rendering
│   ├── rdk_x5_stream_ground_v6.py           # v6: Multi-file GS separation + gas/SOM telemetry wiring
│   ├── rdk_x5_stream_ground_v5.py           # v5: Integrated /perf + /detections + dual conf thresholding
│   ├── rdk_x5_stream_ground_v4.py           # v4: System telemetry (/perf, /proc/meminfo, clocks)
│   ├── rdk_x5_stream_ground_v3.py           # v3: Multi-backend BPU integration (hobot_dnn)
│   ├── rdk_x5_stream_ground_v2.py           # v2: Zero-copy display buffer optimization
│   ├── rdk_x5_stream_ground_v1.py           # v1: Initial MJPEG dual streaming + ONNX CPU inference
│   └── patches/
│       └── apply_clahe_fix.py               # Single-pass CLAHE patch utility
│
├── ground_station/                          # Web Console UI Iterations (HTML & JS)
│   ├── Rover_GS_fix_v1.html                 # [LATEST PRODUCTION v10] Two-file GS console
│   ├── gs_support_v2.js                     # [LATEST PRODUCTION v10] Companion JS telemetry runtime
│   ├── Rover_Ground_Station.dc.html         # 3D Dashboard Source Template (PERF tab & visual effects)
│   ├── Rover_Ground_Station_standalone-src.dc.html # Standalone 3D Dashboard source
│   ├── 3d_console_source/                   # Full 3D Console Suite (screenshots, assets, uploads)
│   ├── Rover_GS_final_v1.html               # v9 Web Console
│   ├── Rover_GS_selfcontained_v1.html       # v8 Self-contained inline console
│   ├── Rover_GS_live_v3.html                # v7 Live Console
│   ├── gs_support_v3.js                     # v7 Support JS
│   ├── Rover_GS_live_v2.html                # v6 Live Console
│   ├── Rover_Ground_Station_live_v1.html    # v4/v5 Live Telemetry Console
│   ├── Rover_Ground_Station_v2.html         # v2 Ground Station Console
│   ├── Rover_Ground_Station.html            # v1 Original Ground Station Console
│   └── support.js                           # v1 Original Support Script
│
├── c_inference/                             # High-Performance C/C++ BPU Engine & Accelerated Postprocess
│   ├── CMakeLists.txt                       # CMake build definition
│   ├── Makefile                             # Standalone build Makefile
│   ├── include/
│   │   ├── bpu_yolo_infer.h                 # BPU model loader & runner interface (Horizon libdnn)
│   │   ├── postprocess.h                    # DFL decoder, Sigmoid LUT, NMS, and colormapping
│   │   └── v4l2_capture.h                   # Lightweight V4L2 zero-copy camera capture interface
│   ├── src/
│   │   ├── bpu_yolo_infer.c                 # BPU tensor allocation, inference submission, output extraction
│   │   ├── postprocess.c                    # Optimized C DFL decode + IoU calculation + NMS
│   │   ├── v4l2_capture.c                   # V4L2 memory-mapped frame capture
│   │   └── main_infer.c                     # Standalone CLI binary for dual BPU inference & benchmarks
│   └── python_binding/
│       ├── bpu_postprocess_ctypes.py        # Python wrapper to call C postprocessing shared library
│       └── libpostprocess.so                # Pre-built shared library
│
├── model_conversion/                        # OpenExplorer BPU Toolchain & Quantization Configs
│   ├── README.md                            # Comprehensive BPU model conversion guide
│   ├── convert_yolo11_to_bpu.sh             # End-to-end conversion script (Docker invocation)
│   ├── yamls/
│   │   ├── thermal_yolo11n_v3_bayese_640x640_nv12.yaml  # Thermal YOLO11n PTQ YAML
│   │   ├── yolo11m_detect_bayese_640x640_nv12.yaml       # RGB YOLO11m PTQ YAML
│   │   ├── yolo11n_detect_bayese_640x640_nv12.yaml       # RGB YOLO11n PTQ YAML
│   │   └── yolov5_detect_bayese_640x640_nv12.yaml        # Reference YOLOv5 YAML
│   └── calibration/
│       └── prepare_calibration_data.py      # Script to generate NV12 calibration image set
│
├── camera_bringup/                          # Camera & Hardware Bringup Scripts
│   ├── rover_cams_v8.py                     # Latest standalone camera test harness (v8)
│   ├── rover_cams_v7.py                     # v7 Camera test script
│   ├── rover_cams_v6.py                     # v6 Camera test script
│   ├── rover_cams_v5.py                     # v5 Camera test script
│   ├── rover_cams_v4.py                     # v4 Camera test script
│   ├── rover_cams_v3.py                     # v3 Camera test script
│   ├── rover_cams_v2.py                     # v2 Camera test script
│   ├── rover_cams.py                        # v1 Camera test script
│   ├── sar_cams.py                          # Search & Rescue camera test script
│   └── setup_imx219_mipi.sh                 # Shell script for cam-service & media-ctl bringup
│
├── diagnostics_and_benchmarks/              # Profiling, Benchmarking & Research Tools
│   ├── bench.py                             # Synthetic pure-inference thread & latency sweeper
│   ├── infer_test.py                        # One-shot camera frame inference tester
│   ├── rdk_x5_dual_infer_headless.py        # Headless dual inference benchmark
│   ├── rdk_x5_dual_infer_lowlat.py          # Low-latency thread-tuned pipeline
│   ├── rdk_x5_stream_autofmt.py             # Auto-formatting stream utility
│   ├── rdk_x5_stream_bpu22.py               # BPU 22ms sync profile test
│   ├── rdk_x5_stream_dualbpu.py             # Dual BPU parallel submission benchmark
│   ├── rdk_x5_stream_native.py              # Native resolution benchmark
│   ├── rdk_x5_stream_sharp.py               # Sharpening kernel filter experiment
│   ├── rdk_x5_stream_tunable.py             # Dynamically tunable pipeline parameters
│   ├── rdk_x5_teleop.py                     # Teleoperation latency measurement harness
│   └── rdk_x5_teleop_v2.py                  # Teleoperation v2 harness
│
└── docs/                                    # Technical Deep-Dive Documentation
    ├── 01_hardware_and_camera_bringup.md    # IMX219 MIPI CSI-2 & Senxor UVC Thermal Bringup
    ├── 02_bpu_model_conversion_guide.md     # OpenExplorer PTQ, YAMLs, Docker, and makertbin
    ├── 03_pipeline_evolution_v1_to_v10.md   # In-depth breakdown of every version iteration
    ├── 04_performance_and_bottleneck_analysis.md # py-spy profiling, CPU vs BPU, 5ms vs 34ms
    ├── 05_c_inference_acceleration.md       # C engine architecture and zero-copy optimization
    └── 06_ground_station_architecture.md    # Web console HUD, MJPEG streams, and telemetry
```

---

## Pipeline Evolution & Version History (v1 - v10)

| Iteration | File Names | Key Architectural Innovation | End-to-End Latency | RGB FPS | Thermal FPS |
|---|---|---|---|---|---|
| **v1** | `rdk_x5_stream_ground_v1.py` / `Rover_Ground_Station.html` | Initial dual-stream HTTP server with ONNX CPU inference | 240 ms | 15–20 | 4–5 |
| **v2** | `rdk_x5_stream_ground_v2.py` / `Rover_Ground_Station_v2.html` | In-place zero-copy frame buffer eliminating ~920 KB copy | 180 ms | 25–30 | 5 |
| **v3** | `rdk_x5_stream_ground_v3.py` | Multi-backend BPU integration (`hobot_dnn` API) | 75 ms | 30 | 13–15 |
| **v4** | `rdk_x5_stream_ground_v4.py` / `Rover_Ground_Station_live_v1.html` | Live system telemetry endpoint (`/perf`, `/sys/class/thermal`) | 72 ms | 30 | 14 |
| **v5** | `rdk_x5_stream_ground_v5.py` | Reintegrated `/detections` JSON, `--overlay`, and dual conf flags | 70 ms | 30 | 14 |
| **v6** | `rdk_x5_stream_ground_v6.py` / `Rover_GS_live_v2.html` + `gs_support_v2.js` | Split GS into modular two-file setup; wired gas & vision canvas | 68 ms | 30 | 15 |
| **v7** | `rdk_x5_stream_ground_v7.py` / `Rover_GS_live_v3.html` | White-hot grayscale ramp + CSS `image-rendering: crisp-edges` | 68 ms | 30 | 15 |
| **v8** | `rdk_x5_stream_ground_v8.py` / `Rover_GS_selfcontained_v1.html` | Attempted single-file inline JavaScript to eliminate 404 routing | *Broke JS* | — | — |
| **v9** | `rdk_x5_stream_ground_v9.py` / `Rover_GS_final_v1.html` | Inlined template clean rollback with corrected gradient | 68 ms | 30 | 15 |
| **v10** | `rdk_x5_stream_ground_v10.py` / `Rover_GS_fix_v1.html` + `gs_support_v2.js` | **[LATEST PRODUCTION]** Modular 2-file GS, live telemetry, dual BPU | **65 ms** | **30** | **15** |

---

## Performance Benchmarks & Profiling

### 1. BPU vs. CPU Latency Breakdown

```
+-------------------------------------------------------------------------+
| Pure BPU Forward Pass (YOLO11n 640x640 INT8):          5.20 ms          |
| Pure BPU Forward Pass (YOLO11m 640x640 INT8):          5.20 ms          |
| Native C Post-Processing (DFL + Sigmoid LUT + NMS):    0.95 ms          |
| Python Post-Processing (NumPy + Interpreted Loops):    34.10 ms (SLOW)  |
+-------------------------------------------------------------------------+
```

### 2. Throughput Capabilities
- **BPU Hardware Limit**: 117.6 FPS (Single Thread) / 163.9 FPS (Dual Thread)
- **Live Production Streamer v10**: 30 FPS RGB / 15 FPS Thermal (Glass-to-glass latency: 65 ms)
- **Native C Standalone Engine**: 133.8 FPS (Dual-modality continuous inference)

---

## Quickstart & Usage Guide

### 1. Hardware Initialization (on RDK X5)
```bash
sudo bash camera_bringup/setup_imx219_mipi.sh
```

### 2. Launch Production Pipeline v10
```bash
cd pipeline
sudo -E python3 rdk_x5_stream_ground_v10.py \
    --thermal-cam 0 \
    --rgb-cam 10 \
    --colormap none \
    --no-hud \
    --overlay \
    --thermal-backend bpu \
    --rgb-backend bpu \
    --thermal-conf 0.5 \
    --rgb-conf 0.35 \
    --port 8080
```

### 3. Open Operator Ground Station
Open any modern web browser and navigate to:
```
http://<RDK_X5_IP_ADDRESS>:8080/
```

### 4. Build and Run Native C Inference Engine
```bash
cd c_inference
make
./build/bpu_infer_cli
```

---

## Technical Documentation Index

For exhaustive technical deep-dives, consult the `docs/` folder:
- [01. Hardware & Dual-Camera Bringup Guide](docs/01_hardware_and_camera_bringup.md)
- [02. BPU Model Conversion & Quantization Guide](docs/02_bpu_model_conversion_guide.md)
- [03. Pipeline Evolution Log (v1 through v10)](docs/03_pipeline_evolution_v1_to_v10.md)
- [04. Performance Profiling & Bottleneck Analysis](docs/04_performance_and_bottleneck_analysis.md)
- [05. High-Performance C/C++ Inference Acceleration](docs/05_c_inference_acceleration.md)
- [06. Ground Station Web Console Architecture](docs/06_ground_station_architecture.md)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
