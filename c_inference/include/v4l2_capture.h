/**
 * @file v4l2_capture.h
 * @brief Zero-copy memory-mapped V4L2 camera capture interface for RDK X5.
 */

#ifndef FYDP_V4L2_CAPTURE_H
#define FYDP_V4L2_CAPTURE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define V4L2_MAX_BUFFERS 4

typedef struct {
    void* start;
    size_t length;
} V4L2Buffer;

typedef struct {
    int fd;
    char dev_name[64];
    int width;
    int height;
    uint32_t pixel_format;
    V4L2Buffer buffers[V4L2_MAX_BUFFERS];
    int buffer_count;
    bool is_streaming;
} V4L2Camera;

/**
 * @brief Initializes and opens a V4L2 camera device with memory mapping.
 */
int v4l2_camera_open(V4L2Camera* cam, const char* dev_path, int width, int height, uint32_t pixel_format);

/**
 * @brief Starts video stream capture.
 */
int v4l2_camera_start(V4L2Camera* cam);

/**
 * @brief Reads a single zero-copy frame buffer.
 */
int v4l2_camera_get_frame(V4L2Camera* cam, const void** frame_data, size_t* frame_size, int* buf_index);

/**
 * @brief Returns the frame buffer back to the kernel queue.
 */
int v4l2_camera_release_frame(V4L2Camera* cam, int buf_index);

/**
 * @brief Stops video stream and unmaps memory.
 */
void v4l2_camera_close(V4L2Camera* cam);

#ifdef __cplusplus
}
#endif

#endif /* FYDP_V4L2_CAPTURE_H */
