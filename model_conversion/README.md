# BPU Model Conversion Guide — YOLO11 to Horizon Bayes-e

This guide describes how to convert PyTorch YOLO11 detection models into optimized `.bin` models running on the D-Robotics Horizon RDK X5 Brain Processing Unit (BPU).

---

## 1. Overview & Architecture

- **Target Architecture**: Horizon Bayes-e BPU Core (RDK X5 4GB)
- **Input Format**: NV12 (YUV420SP) 640×640
- **Quantization Strategy**: Post-Training Quantization (PTQ) INT8 with calibration dataset
- **Toolchain Environment**: x86_64 Host running Docker image `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8`

---

## 2. Step-by-Step Conversion Pipeline

### Step 1: Export PyTorch Model to ONNX

Export the trained PyTorch weights (`.pt`) to ONNX using Ultralytics with DFL outputs preserved:

```bash
python3 export.py --weights thermal_yolo11n_v3_best.pt --imgsz 640 --format onnx --opset 11
```

The exported ONNX model produces 6 tensor outputs (3 detection scales with separated bbox regression and class scores):
- `(1, 80, 80, 1)`, `(1, 80, 80, 64)` (80×80 stride 8)
- `(1, 40, 40, 1)`, `(1, 40, 40, 64)` (40×40 stride 16)
- `(1, 20, 20, 1)`, `(1, 20, 20, 64)` (20×20 stride 32)

### Step 2: Prepare Calibration Dataset

Run the calibration preparation script to generate NV12 binary images:

```bash
python3 calibration/prepare_calibration_data.py --src ./raw_images --dst ./cal --thermal --count 100
```

### Step 3: Run OpenExplorer BPU Compiler

Execute `hb_mapper makertbin` using the automated script:

```bash
bash convert_yolo11_to_bpu.sh yamls/thermal_yolo11n_v3_bayese_640x640_nv12.yaml
```

The resulting `.bin` file (`thermal_yolo11n_v3_bayese_640x640_nv12.bin`) can be transferred directly to the RDK X5 via SCP.
