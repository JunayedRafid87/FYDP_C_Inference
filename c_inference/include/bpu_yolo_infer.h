/**
 * @file bpu_yolo_infer.h
 * @brief Horizon BPU C Runtime interface for YOLO11 inference on RDK X5.
 */

#ifndef FYDP_BPU_YOLO_INFER_H
#define FYDP_BPU_YOLO_INFER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "postprocess.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    char model_path[256];
    int input_h;
    int input_w;
    int num_classes;
    float conf_threshold;
    float iou_threshold;
    int bpu_core_id;
    void* bpu_handle;       /* Horizon dnn handle */
    bool is_initialized;
} BPUModelContext;

/**
 * @brief Loads a compiled .bin model onto the Horizon BPU.
 * 
 * @param ctx Pointer to model context struct
 * @param model_path Path to compiled .bin model file
 * @param num_classes Number of detection classes (1 for thermal, 80 for COCO RGB)
 * @param conf_threshold Confidence threshold
 * @param iou_threshold NMS IoU threshold
 * @param bpu_core_id Target BPU core (0 or 1)
 * @return 0 on success, non-zero error code on failure
 */
int bpu_yolo_init(
    BPUModelContext* ctx,
    const char* model_path,
    int num_classes,
    float conf_threshold,
    float iou_threshold,
    int bpu_core_id
);

/**
 * @brief Submits an NV12 frame to the BPU and runs accelerated inference + post-processing.
 * 
 * @param ctx Initialized model context
 * @param nv12_data Pointer to raw NV12 memory buffer (H * 3/2 * W bytes)
 * @param result Output detection results
 * @return 0 on success, non-zero error code on failure
 */
int bpu_yolo_forward(
    BPUModelContext* ctx,
    const uint8_t* nv12_data,
    DetectionResult* result
);

/**
 * @brief Releases BPU tensors and model memory.
 * 
 * @param ctx Pointer to model context
 */
void bpu_yolo_release(BPUModelContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* FYDP_BPU_YOLO_INFER_H */
