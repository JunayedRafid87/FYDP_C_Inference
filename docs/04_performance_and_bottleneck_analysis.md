# 04. Performance Profiling & Bottleneck Analysis

This document provides an in-depth empirical analysis of system performance, hardware utilization, profiling data, and architectural bottlenecks identified during development.

---

## 1. The BPU Clock vs. Utilization Paradox

A critical observation during testing was:
> *Why does the BPU report 1000 MHz (maximum frequency) while BPU utilization is reported at only 24%–26%?*

### Explanation:
- **Frequency Governor**: The Horizon BPU clock governor locks to maximum frequency (1000 MHz) as soon as any active inferencing session is created to prevent frequency ramp-up latency penalties.
- **Duty Cycle vs Clock**: The BPU forward pass for YOLO11n takes only **~5.2 ms**. At 30 FPS, the BPU is actively executing matrix operations for `30 × 5.2 ms = 156 ms` out of every 1000 ms (~15.6% duty cycle per model). Running two models concurrently yields `~24%–26%` active hardware utilization.
- The BPU is **not saturated**; it completes its computation rapidly and sits idle waiting for the next frame.

---

## 2. Latency Breakdown: 5ms BPU vs. 34ms CPU Post-Processing

Profiling the live Python pipeline with `py-spy` revealed the primary architectural bottleneck:

```
+-------------------------------------------------------------------------+
| TOTAL FRAME TIME IN PYTHON: ~45 ms                                      |
+--------------------+----------------------------------------------------+
| BPU Forward (5 ms) | Python CPU Post-Processing & NMS (34 ms)  | Enc (6)|
+--------------------+----------------------------------------------------+
```

### Breakdown of CPU Overhead:
1. **DFL Tensor Reshaping & Slicing**: Python NumPy array manipulation across 6 output feature maps (`(80,80,64)`, `(40,40,64)`, `(20,20,64)`).
2. **Softmax & Sigmoid Math**: Performed in interpreted Python loops without hardware SIMD acceleration.
3. **Multi-Class NMS**: High bounding box candidate counts in unoptimized Python loops.
4. **JPEG Compression**: CPU-based OpenCV `cv2.imencode` consuming ~6 ms per frame.

---

## 3. BPU Ceiling Benchmarks

Synthetic BPU throughput measurements (excluding Python GIL and camera capture):

| Configuration | Latency | Single-Model Throughput |
|---|---|---|
| Single Thread (`bpu_cores=[0]`) | 8.5 ms | **117.6 FPS** |
| Dual Thread (`bpu_cores=[0, 1]`) | 6.1 ms | **163.9 FPS** |
| Live Python Pipeline v10 | 65.0 ms end-to-end | **30 FPS RGB / 15 FPS Thermal** |
| Native C Pipeline (`c_inference`) | 11.3 ms end-to-end | **133.8 FPS (Dual Stream)** |

---

## 4. Profiling with `py-spy`

To profile the live production streamer on the RDK X5:

```bash
# Identify streamer PID (filter out sudo wrapper)
pgrep -af rdk_x5_stream_ground_v10

# Run live top-level profiler
sudo py-spy top --pid <PID> --nonblocking

# Generate interactive SVG flame graph
sudo py-spy record -o flamegraph.svg --pid <PID> --duration 30
```

---

## 5. Thermal and Power Characteristics

Monitored via `sudo hrut_somstatus`:
- **Ambient Temperature**: 27°C
- **DDR Temperature**: ~65.7°C
- **BPU Temperature**: ~65.0°C
- **CPU Temperature**: ~64.8°C
- **CPU Governor**: Cores 0–7 scaling dynamically between 1200 MHz and 1500 MHz without thermal throttling.
