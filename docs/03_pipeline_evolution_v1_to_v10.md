# 03. Pipeline Evolution Log (v1 through v10)

This document provides a comprehensive chronological record of every pipeline iteration developed during the project, documenting the motivation, code changes, performance deltas, and lessons learned up to **v10 (the latest production release)**.

---

## Evolution Summary Matrix

| Version | Script Name | Ground Station UI | Key Innovation / Major Change | End-to-End Latency | RGB FPS | Thermal FPS |
|---|---|---|---|---|---|---|
| **v1** | `rdk_x5_stream_ground_v1.py` | `Rover_Ground_Station.html` | Initial baseline MJPEG server + ONNX CPU inference | ~240 ms | 15–20 | 4–5 |
| **v2** | `rdk_x5_stream_ground_v2.py` | `Rover_Ground_Station_v2.html` | Removed full-frame `display.copy()` (~920 KB memcpy eliminated) | ~180 ms | 25–30 | 5 |
| **v3** | `rdk_x5_stream_ground_v3.py` | `Rover_Ground_Station_v2.html` | Integrated `hobot_dnn` BPU runtime for both Thermal & RGB | ~75 ms | 30 | 13–15 |
| **v4** | `rdk_x5_stream_ground_v4.py` | `Rover_Ground_Station_live_v1.html` | Added live `/perf` telemetry endpoint (SOM temps, clocks, RAM) | ~72 ms | 30 | 14 |
| **v5** | `rdk_x5_stream_ground_v5.py` | `Rover_Ground_Station_live_v1.html` | Re-added `/detections` JSON, `--overlay`, and dual conf flags | ~70 ms | 30 | 14 |
| **v6** | `rdk_x5_stream_ground_v6.py` | `Rover_GS_live_v2.html` + `gs_support_v2.js` | Split GS into modular two-file setup; wired gas & dynamic vision | ~68 ms | 30 | 15 |
| **v7** | `rdk_x5_stream_ground_v7.py` | `Rover_GS_live_v3.html` + `gs_support_v3.js` | White-hot grayscale ramp + CSS `image-rendering: crisp-edges` | ~68 ms | 30 | 15 |
| **v8** | `rdk_x5_stream_ground_v8.py` | `Rover_GS_selfcontained_v1.html` | Attempted single-file inline JavaScript to eliminate 404 routing | *Broke JS* | — | — |
| **v9** | `rdk_x5_stream_ground_v9.py` | `Rover_GS_final_v1.html` | Inlined template clean rollback with corrected gradient | ~68 ms | 30 | 15 |
| **v10** | `rdk_x5_stream_ground_v10.py` | `Rover_GS_fix_v1.html` + `gs_support_v2.js` | **[LATEST STABLE]** Modular 2-file GS, live telemetry, dual BPU | **~65 ms** | **30** | **15** |

---

## Detailed Version Breakdown

### Version 1 (`rdk_x5_stream_ground_v1.py` / `rdk_x5_stream_ground.py`)
- **Architecture**: Initial dual-stream HTTP server using Python `http.server.ThreadingHTTPServer`. Thermal and RGB streams captured via OpenCV `cv2.VideoCapture` and inferenced using ONNX Runtime on CPU.
- **Bottlenecks**: ONNX CPU inference consumed 100% of 4 CPU cores, dropping thermal inference rate to 4 Hz and causing severe video streaming stutter (>200 ms latency).

### Version 2 (`rdk_x5_stream_ground_v2.py`)
- **Changes**: Replaced full-frame `display.copy()` (~920 KB memcpy per frame at 30 Hz = ~27 MB/s memory bandwidth overhead) with an in-place zero-copy frame buffer.
- **Impact**: RGB streaming reached smooth 30 FPS.

### Version 3 (`rdk_x5_stream_ground_v3.py`)
- **Changes**: Introduced `--thermal-backend bpu` and `--rgb-backend bpu` flags utilizing the Horizon `hobot_dnn` Python API to offload model execution to the BPU. Added `--colormap none` (white-hot) and `--no-hud` options.
- **Impact**: Inference time dropped from 180 ms (CPU) to 5.2 ms (BPU), jumping thermal inference from 4 Hz to 13–15 Hz.

### Version 4 (`rdk_x5_stream_ground_v4.py`)
- **Changes**: Added background `PerfMonitor` thread polling `/sys/class/thermal/`, `/proc/meminfo`, and CPU governor clock frequencies. Added `/perf` HTTP REST route serving real-time telemetry JSON.
- **Impact**: Enabled live hardware monitoring on the Ground Station console.

### Version 5 (`rdk_x5_stream_ground_v5.py`)
- **Changes**: Reintegrated `--overlay` for client-side canvas box rendering, `/detections` JSON endpoint, and distinct `--thermal-conf` and `--rgb-conf` CLI arguments.

### Version 6 (`rdk_x5_stream_ground_v6.py`)
- **Changes**: Modularized web console into `Rover_GS_live_v2.html` and `gs_support_v2.js`. Connected dynamic bounding box overlay to web canvas and hooked up `GasReader` UART telemetry.

### Version 7 (`rdk_x5_stream_ground_v7.py`)
- **Changes**: Switched thermal palette to a true white-to-black grayscale scale matching `--colormap none`. Applied CSS `image-rendering: crisp-edges` to preserve sharp pixel boundaries of the 256×192 thermal sensor.

### Version 8 (`rdk_x5_stream_ground_v8.py`)
- **Changes**: Attempted inlining all JavaScript into a single HTML file to prevent static file 404 errors.
- **Observation**: Template interpolation failed due to inline script parsing conflicts (`{{ item.label }}` raw tokens rendered).

### Version 9 (`rdk_x5_stream_ground_v9.py`)
- **Changes**: Cleaned up template tags and applied gradient corrections.

### Version 10 (`rdk_x5_stream_ground_v10.py`) — **The Latest Production Release**
- **Changes**: Restored the verified two-file modular architecture (`Rover_GS_fix_v1.html` + `gs_support_v2.js`).
- **Features**:
  1. Simultaneous Dual BPU Inference (RGB YOLO11m @ 30 FPS, Thermal YOLO11n @ 15 FPS).
  2. Sub-70ms end-to-end glass-to-glass latency over 5 GHz Wi-Fi / Gigabit Ethernet.
  3. Real-time telemetry endpoints (`/perf`, `/detections`, `/gas`, `/stream_thermal`, `/stream_rgb`).
  4. Crisp 256×192 thermal rendering without browser interpolation blur.
  5. Dedicated single-pass CLAHE contrast enhancement for thermal LWIR imagery.
