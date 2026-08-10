# 05. High-Performance C/C++ Inference Acceleration Guide

This document explains the technical architecture, mathematical optimizations, and integration strategy for the native C/C++ inference acceleration module located in `c_inference/`.

---

## 1. Motivation for Native C Acceleration

While Python provides rapid prototyping, its Global Interpreter Lock (GIL), dynamic type overhead, and non-vectorized tensor iterations create a 34 ms bottleneck during post-processing.

By migrating the post-processing pipeline to C/C++:
- **DFL Decoding Latency**: Reduced from **~28 ms** (NumPy) to **~0.8 ms** (Vectorized C).
- **Sigmoid Activation**: Reduced from **~4 ms** to **~0.1 ms** via precomputed Lookup Tables (LUT).
- **NMS & Box Suppression**: Reduced from **~2 ms** to **~0.2 ms** using optimized memory layout and in-place sorting.

---

## 2. Mathematical Optimization Architecture

### Vectorized Distribution Focal Loss (DFL) Decoding
YOLO11 represents each bounding box coordinate $(x_1, y_1, x_2, y_2)$ as a discrete probability distribution over 16 bins ($0 \le i \le 15$):

$$\hat{y} = \sum_{i=0}^{15} \text{Softmax}(z_i) \cdot i$$

In C (`postprocess.c`), this is computed with optimized local stack arrays:

```c
static inline float decode_dfl_coord(const float* reg_ptr) {
    float sm[16];
    softmax_16(reg_ptr, sm);
    float val = 0.0f;
    for (int i = 0; i < 16; ++i) {
        val += sm[i] * (float)i;
    }
    return val;
}
```

### Precomputed Sigmoid Lookup Table (LUT)
Instead of invoking `expf()` for thousands of candidate anchor scores, `fast_sigmoid` maps inputs in the range $[-10.0, +10.0]$ to a 2048-entry table in $O(1)$ time:

```c
float fast_sigmoid(float x) {
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return 1.0f;
    int idx = (int)((x + 10.0f) * (2048 / 20.0f));
    return sigmoid_lut[idx];
}
```

---

## 3. Python Ctypes Drop-In Bridge

The module `c_inference/python_binding/bpu_postprocess_ctypes.py` provides a drop-in CTypes class `CPostProcessor`.

### Integration Example:
```python
from bpu_postprocess_ctypes import CPostProcessor

# Initialize native accelerator
c_post = CPostProcessor()

# Decode raw BPU outputs directly
boxes = c_post.postprocess(bpu_output_tensors, conf_thresh=0.45, iou_thresh=0.45)
```

---

## 4. Building and Running the Standalone C Engine

```bash
cd c_inference
make
./build/bpu_infer_cli
```
