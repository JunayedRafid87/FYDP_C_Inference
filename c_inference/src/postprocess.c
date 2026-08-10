/**
 * @file postprocess.c
 * @brief High-performance C implementation of YOLO11 DFL decoding and NMS.
 */

#include "postprocess.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float sigmoid_lut[SIGMOID_LUT_SIZE];
static bool luts_initialized = false;

#define SIGMOID_RANGE 10.0f

void init_postprocess_luts(void) {
    if (luts_initialized) return;
    for (int i = 0; i < SIGMOID_LUT_SIZE; ++i) {
        float x = -SIGMOID_RANGE + (2.0f * SIGMOID_RANGE * i) / (float)SIGMOID_LUT_SIZE;
        sigmoid_lut[i] = 1.0f / (1.0f + expf(-x));
    }
    luts_initialized = true;
}

float fast_sigmoid(float x) {
    if (!luts_initialized) init_postprocess_luts();
    if (x <= -SIGMOID_RANGE) return 0.0f;
    if (x >= SIGMOID_RANGE) return 1.0f;
    int idx = (int)((x + SIGMOID_RANGE) * (SIGMOID_LUT_SIZE / (2.0f * SIGMOID_RANGE)));
    if (idx < 0) idx = 0;
    if (idx >= SIGMOID_LUT_SIZE) idx = SIGMOID_LUT_SIZE - 1;
    return sigmoid_lut[idx];
}

static inline void softmax_16(const float* src, float* dst) {
    float max_v = src[0];
    for (int i = 1; i < 16; ++i) {
        if (src[i] > max_v) max_v = src[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
        dst[i] = expf(src[i] - max_v);
        sum += dst[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < 16; ++i) {
        dst[i] *= inv_sum;
    }
}

static inline float decode_dfl_coord(const float* reg_ptr) {
    float sm[16];
    softmax_16(reg_ptr, sm);
    float val = 0.0f;
    for (int i = 0; i < 16; ++i) {
        val += sm[i] * (float)i;
    }
    return val;
}

float compute_iou(const DetectionBox* a, const DetectionBox* b) {
    float x1 = (a->x1 > b->x1) ? a->x1 : b->x1;
    float y1 = (a->y1 > b->y1) ? a->y1 : b->y1;
    float x2 = (a->x2 < b->x2) ? a->x2 : b->x2;
    float y2 = (a->y2 < b->y2) ? a->y2 : b->y2;

    float inter_w = x2 - x1;
    float inter_h = y2 - y1;
    if (inter_w <= 0.0f || inter_h <= 0.0f) return 0.0f;

    float inter_area = inter_w * inter_h;
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);

    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

static int compare_boxes(const void* a, const void* b) {
    const DetectionBox* box_a = (const DetectionBox*)a;
    const DetectionBox* box_b = (const DetectionBox*)b;
    if (box_b->score > box_a->score) return 1;
    if (box_b->score < box_a->score) return -1;
    return 0;
}

void run_nms(DetectionBox* candidates, int num_candidates, float iou_threshold, DetectionResult* result) {
    if (num_candidates <= 0) {
        result->count = 0;
        return;
    }

    qsort(candidates, num_candidates, sizeof(DetectionBox), compare_boxes);

    uint8_t* suppressed = (uint8_t*)calloc(num_candidates, sizeof(uint8_t));
    int count = 0;

    for (int i = 0; i < num_candidates && count < MAX_DETECTIONS; ++i) {
        if (suppressed[i]) continue;

        result->boxes[count++] = candidates[i];

        for (int j = i + 1; j < num_candidates; ++j) {
            if (suppressed[j]) continue;
            if (candidates[i].class_id == candidates[j].class_id) {
                float iou = compute_iou(&candidates[i], &candidates[j]);
                if (iou > iou_threshold) {
                    suppressed[j] = 1;
                }
            }
        }
    }

    free(suppressed);
    result->count = count;
}

void decode_yolo11_outputs(
    const FeatureScale* scales,
    int num_scales,
    float conf_threshold,
    float iou_threshold,
    int target_w,
    int target_h,
    DetectionResult* result
) {
    if (!luts_initialized) init_postprocess_luts();

    DetectionBox* candidate_buffer = (DetectionBox*)malloc(sizeof(DetectionBox) * 10000);
    int num_candidates = 0;

    for (int s = 0; s < num_scales; ++s) {
        const FeatureScale* sc = &scales[s];
        int grid_h = sc->grid_h;
        int grid_w = sc->grid_w;
        int stride = sc->stride;
        int num_classes = sc->num_classes;

        const float* cls_base = sc->cls_ptr;
        const float* reg_base = sc->reg_ptr;

        for (int h = 0; h < grid_h; ++h) {
            for (int w = 0; w < grid_w; ++w) {
                int grid_idx = h * grid_w + w;
                const float* cls_p = cls_base + grid_idx * num_classes;
                const float* reg_p = reg_base + grid_idx * 64;

                // Find max class confidence
                int best_cls = 0;
                float best_score = cls_p[0];
                for (int c = 1; c < num_classes; ++c) {
                    if (cls_p[c] > best_score) {
                        best_score = cls_p[c];
                        best_cls = c;
                    }
                }

                float conf = fast_sigmoid(best_score);
                if (conf < conf_threshold) continue;

                // DFL Decode 4 bounding box offsets
                float left   = decode_dfl_coord(reg_p + 0);
                float top    = decode_dfl_coord(reg_p + 16);
                float right  = decode_dfl_coord(reg_p + 32);
                float bottom = decode_dfl_coord(reg_p + 48);

                float center_x = ((float)w + 0.5f) * (float)stride;
                float center_y = ((float)h + 0.5f) * (float)stride;

                float x1 = (center_x - left * (float)stride);
                float y1 = (center_y - top * (float)stride);
                float x2 = (center_x + right * (float)stride);
                float y2 = (center_y + bottom * (float)stride);

                // Clip to target image dimensions
                if (x1 < 0.0f) x1 = 0.0f;
                if (y1 < 0.0f) y1 = 0.0f;
                if (x2 > (float)target_w) x2 = (float)target_w;
                if (y2 > (float)target_h) y2 = (float)target_h;

                if (num_candidates < 10000) {
                    DetectionBox* box = &candidate_buffer[num_candidates++];
                    box->x1 = x1;
                    box->y1 = y1;
                    box->x2 = x2;
                    box->y2 = y2;
                    box->score = conf;
                    box->class_id = best_cls;
                }
            }
        }
    }

    run_nms(candidate_buffer, num_candidates, iou_threshold, result);
    free(candidate_buffer);
}

void apply_white_hot_colormap(const uint8_t* src_gray, uint8_t* dst_bgr, int width, int height) {
    int total_pixels = width * height;
    for (int i = 0; i < total_pixels; ++i) {
        uint8_t val = src_gray[i];
        dst_bgr[i * 3 + 0] = val; /* B */
        dst_bgr[i * 3 + 1] = val; /* G */
        dst_bgr[i * 3 + 2] = val; /* R */
    }
}
