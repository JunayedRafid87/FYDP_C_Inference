# 06. Ground Station Web Console Architecture

This document describes the design, protocols, and rendering mechanics of the Ground Station web console (`Rover_GS_fix_v1.html` + `gs_support_v2.js`).

---

## 1. Architectural Overview

The Ground Station console provides a browser-based Command and Control (C2) dashboard for rover teleoperation and sensor monitoring:

```
+-------------------------------------------------------------------------+
|                  GROUND STATION DASHBOARD (Port 8080)                   |
+------------------------------------+------------------------------------+
|  PRIMARY RGB TELEOP FEED (640x480) |  LWIR THERMAL FEED (256x192)       |
|  - Live MJPEG Stream (30 FPS)      |  - White-Hot Grayscale (15 FPS)    |
|  - AI Bounding Box Canvas Overlay  |  - Pixel-Crisp Sensor Rendering    |
+------------------------------------+------------------------------------+
|  ATMOSPHERE & GAS TELEMETRY        |  HARDWARE & BPU METRICS (/perf)    |
|  - MQ-Sensor Gas Concentrations    |  - Core Clocks: 1200 / 1500 MHz    |
|  - Threat Level Warning Indicator  |  - SOM Temps: DDR/BPU/CPU ~65°C    |
|  - Live Historical Graph           |  - RAM: 352 / 3062 MB              |
+------------------------------------+------------------------------------+
```

---

## 2. HTTP Streaming & REST Telemetry Routes

The backend server implements non-blocking, multi-threaded request routing:

| Endpoint | Method | Payload Type | Description |
|---|---|---|---|
| `/` | `GET` | `text/html` | Serves the main Ground Station HTML console |
| `/gs_support_v2.js` | `GET` | `application/javascript` | Serves the companion telemetry runtime |
| `/stream_rgb` | `GET` | `multipart/x-mixed-replace` | Live MJPEG video stream from Sony IMX219 (30 FPS) |
| `/stream_thermal` | `GET` | `multipart/x-mixed-replace` | Live MJPEG video stream from Senxor Thermal (15 FPS) |
| `/perf` | `GET` | `application/json` | Real-time board hardware telemetry (temps, clocks, RAM, load) |
| `/detections` | `GET` | `application/json` | Current bounding boxes, classes, and confidence scores |
| `/gas` | `GET` | `application/json` | Live atmospheric gas concentration levels |

---

## 3. High-Fidelity Rendering Mechanics

### 1. `image-rendering: crisp-edges`
Because the thermal sensor captures native 256×192 frames, default browser bilinear filtering blurs pixel boundaries. Setting `image-rendering: crisp-edges` renders individual LWIR sensor pixels sharply without artificial smoothing.

### 2. Client-Side Bounding Box Canvas Overlay
Instead of burning bounding boxes directly into JPEG video frames on the rover CPU (which consumes memory bandwidth and prevents raw telemetry recording), bounding boxes are sent as lightweight JSON over `/detections` and rendered client-side on an HTML5 `<canvas>` positioned over the video stream.
