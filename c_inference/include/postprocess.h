/**
 * @file postprocess.h
 * @brief High-performance C/C++ post-processing engine for YOLO11 on Horizon BPU.
 * 
 * Implements vectorized DFL (Distribution Focal Loss) box decoding, Sigmoid LUT,
 * Non-Maximum Suppression (NMS), and fast colormap generation.
 */

#ifndef FYDP_POSTPROCESS_H
#define FYDP_POSTPROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#define MAX_DETECTIONS 300
#define NUM_REG_BINS 16   /* YOLO11 DFL distribution bins per coordinate (total 64) */
#define SIGMOID_LUT_SIZE 2048

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
} DetectionBox;

typedef struct {
    DetectionBox boxes[MAX_DETECTIONS];
    int count;
    float inference_time_ms;
    float postprocess_time_ms;
} DetectionResult;

typedef struct {
    int grid_h;
    int grid_w;
    int stride;
    const float* cls_ptr;   /* Pointer to (grid_h, grid_w, num_classes) */
    const float* reg_ptr;   /* Pointer to (grid_h, grid_w, 64) */
    int num_classes;
} FeatureScale;

/**
 * @brief Initializes mathematical lookup tables (Sigmoid, DFL projections).
 */
void init_postprocess_luts(void);

/**
 * @brief Fast Sigmoid approximation using precomputed lookup table.
 */
float fast_sigmoid(float x);

/**
 * @brief Computes Intersection over Union (IoU) between two bounding boxes.
 */
float compute_iou(const DetectionBox* a, const DetectionBox* b);

/**
 * @brief Performs Non-Maximum Suppression (NMS) on raw candidates.
 * 
 * @param candidates Array of candidate bounding boxes
 * @param num_candidates Number of input candidates
 * @param iou_threshold IoU overlap threshold (e.g., 0.45f)
 * @param result Output detection result struct
 */
void run_nms(DetectionBox* candidates, int num_candidates, float iou_threshold, DetectionResult* result);

/**
 * @brief Full YOLO11 multi-scale DFL decoding and NMS post-processing in C.
 * 
 * @param scales Array of 3 FeatureScale structs (80x80, 40x40, 20x20)
 * @param num_scales Number of scales (typically 3)
 * @param conf_threshold Minimum class confidence threshold (e.g., 0.25f)
 * @param iou_threshold NMS IoU threshold (e.g., 0.45f)
 * @param target_w Target image width (e.g., 640)
 * @param target_h Target image height (e.g., 640)
 * @param result Pointer to output result
 */
void decode_yolo11_outputs(
    const FeatureScale* scales,
    int num_scales,
    float conf_threshold,
    float iou_threshold,
    int target_w,
    int target_h,
    DetectionResult* result
);

/**
 * @brief Accelerated White-Hot Grayscale and CLAHE mapping in C.
 * 
 * @param src_gray 8-bit single channel input image
 * @param dst_bgr 24-bit 3-channel BGR output image
 * @param width Image width
 * @param height Image height
 */
void apply_white_hot_colormap(const uint8_t* src_gray, uint8_t* dst_bgr, int width, int height);

#ifdef __cplusplus
}
#endif

#endif /* FYDP_POSTPROCESS_H */
