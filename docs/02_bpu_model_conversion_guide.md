# 02. BPU Model Conversion & Quantization Guide

This document details the complete process of converting Ultralytics YOLO11 models from PyTorch into quantized `.bin` binaries targeting the **Horizon Bayes-e Brain Processing Unit (BPU)**.

---

## 1. Overview of the Horizon BPU Architecture

The Horizon BPU (Bayes-e) on the RDK X5 provides high-throughput tensor acceleration with hardware-supported INT8 and INT16 precision.

- **Peak Compute**: 10 TOPS INT8
- **Native Memory Layout**: NV12 (YUV420SP) and NHWC tensor format
- **BPU Forward Latency**: ~5.2 ms for YOLO11n (640×640), ~7.8 ms for YOLO11m (640×640)

---

## 2. Docker Toolchain Setup (x86_64 Host)

The Horizon OpenExplorer AI Toolchain runs inside a Docker container on an x86_64 development workstation:

```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

---

## 3. ONNX Model Export

Export the trained PyTorch checkpoint with separated DFL heads:

```bash
python3 export.py \
    --weights thermal_yolo11n_v3_best.pt \
    --imgsz 640 \
    --format onnx \
    --opset 11 \
    --simplify
```

The exported graph exposes 6 multi-scale feature tensors:
- Stride 8: `(1, 80, 80, num_classes)` + `(1, 80, 80, 64)`
- Stride 16: `(1, 40, 40, num_classes)` + `(1, 40, 40, 64)`
- Stride 32: `(1, 20, 20, num_classes)` + `(1, 20, 20, 64)`

---

## 4. Post-Training Quantization (PTQ) Configuration

The YAML configuration controls graph optimization and calibration:

```yaml
model_parameters:
  onnx_model: 'thermal_yolo11n_v3_best.onnx'
  output_model_file_prefix: 'thermal_yolo11n_v3_bayese_640x640_nv12'
  march: 'bayes-e'

input_parameters:
  input_name: 'images'
  input_type_train: 'rgb'
  input_type_rt: 'nv12'
  input_layout_train: 'NCHW'
  input_layout_rt: 'NHWC'
  input_shape: '1x3x640x640'
  norm_type: 'data_scale'
  scale_value: [0.003921568627451, 0.003921568627451, 0.003921568627451]

calibration_parameters:
  cal_data_dir: './cal'
  cal_data_type: 'float32'
  calibration_type: 'default'
  max_percentile: 0.9999

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O3'
  core_num: 1
```

---

## 5. Model Compilation (`hb_mapper makertbin`)

Execute the compiler inside the container:

```bash
docker run --rm -v "$(pwd):/workspace" -w /workspace \
    openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 \
    hb_mapper makertbin --config yamls/thermal_yolo11n_v3_bayese_640x640_nv12.yaml --model-type onnx
```

The output file `thermal_yolo11n_v3_bayese_640x640_nv12.bin` contains the compiled hardware instructions and quantized weights ready for BPU deployment.
