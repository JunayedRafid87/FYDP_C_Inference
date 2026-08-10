/**
 * @file main_infer.c
 * @brief Standalone CLI benchmarking and inference harness for Horizon BPU.
 */

#include "bpu_yolo_infer.h"
#include "postprocess.h"
#include "v4l2_capture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

int main(int argc, char** argv) {
    printf("==============================================================================\n");
    printf(" FYDP_C_Inference - High Performance BPU Engine (D-Robotics Horizon RDK X5)\n");
    printf("==============================================================================\n\n");

    const char* thermal_model = (argc > 1) ? argv[1] : "thermal_yolo11n_v3_bayese_640x640_nv12.bin";
    const char* rgb_model     = (argc > 2) ? argv[2] : "yolo11m_detect_bayese_640x640_nv12.bin";
    int iterations = (argc > 3) ? atoi(argv[3]) : 100;

    printf("[+] Target Models:\n");
    printf("    Thermal Model: %s\n", thermal_model);
    printf("    RGB Model:     %s\n", rgb_model);
    printf("    Benchmark Iterations: %d\n\n", iterations);

    // Initialize BPU contexts
    BPUModelContext ctx_thermal;
    BPUModelContext ctx_rgb;

    bpu_yolo_init(&ctx_thermal, thermal_model, 1, 0.50f, 0.45f, 0);
    bpu_yolo_init(&ctx_rgb, rgb_model, 80, 0.35f, 0.45f, 0);

    // Allocate synthetic 640x640 NV12 buffer (640 * 640 * 1.5 = 614,400 bytes)
    size_t nv12_size = 640 * 640 * 3 / 2;
    uint8_t* nv12_buf = (uint8_t*)malloc(nv12_size);
    memset(nv12_buf, 128, nv12_size);

    printf("[+] Running Dual-Modality Inference Benchmark (%d iterations)...\n", iterations);

    double total_thermal_bpu = 0.0;
    double total_thermal_post = 0.0;
    double total_rgb_bpu = 0.0;
    double total_rgb_post = 0.0;

    DetectionResult res_thermal;
    DetectionResult res_rgb;

    double t_start = get_time_ms();

    for (int i = 0; i < iterations; ++i) {
        bpu_yolo_forward(&ctx_thermal, nv12_buf, &res_thermal);
        total_thermal_bpu  += res_thermal.inference_time_ms;
        total_thermal_post += res_thermal.postprocess_time_ms;

        bpu_yolo_forward(&ctx_rgb, nv12_buf, &res_rgb);
        total_rgb_bpu  += res_rgb.inference_time_ms;
        total_rgb_post += res_rgb.postprocess_time_ms;
    }

    double t_end = get_time_ms();
    double total_wall_time = t_end - t_start;

    printf("\n==============================================================================\n");
    printf(" BENCHMARK RESULTS SUMMARY\n");
    printf("==============================================================================\n");
    printf(" Thermal Model (YOLO11n 640x640):\n");
    printf("   - Avg BPU Forward Latency:       %6.2f ms\n", total_thermal_bpu / iterations);
    printf("   - Avg C Postprocess Latency:     %6.2f ms\n", total_thermal_post / iterations);
    printf("   - Total Avg Latency:             %6.2f ms\n", (total_thermal_bpu + total_thermal_post) / iterations);
    printf("   - Detections Found:              %d\n\n", res_thermal.count);

    printf(" RGB Model (YOLO11m 640x640):\n");
    printf("   - Avg BPU Forward Latency:       %6.2f ms\n", total_rgb_bpu / iterations);
    printf("   - Avg C Postprocess Latency:     %6.2f ms\n", total_rgb_post / iterations);
    printf("   - Total Avg Latency:             %6.2f ms\n", (total_rgb_bpu + total_rgb_post) / iterations);
    printf("   - Detections Found:              %d\n\n", res_rgb.count);

    printf(" Throughput & Performance:\n");
    printf("   - Combined Wall Time:            %6.2f ms for %d dual iterations\n", total_wall_time, iterations);
    printf("   - Combined Throughput:           %6.2f FPS (Dual Modality)\n", (iterations * 2.0 * 1000.0) / total_wall_time);
    printf("==============================================================================\n");

    free(nv12_buf);
    bpu_yolo_release(&ctx_thermal);
    bpu_yolo_release(&ctx_rgb);

    return 0;
}
