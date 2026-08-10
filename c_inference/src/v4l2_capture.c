/**
 * @file v4l2_capture.c
 * @brief Memory-mapped V4L2 zero-copy camera capture implementation.
 */

#include "v4l2_capture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

static int xioctl(int fh, int request, void *arg) {
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

int v4l2_camera_open(V4L2Camera* cam, const char* dev_path, int width, int height, uint32_t pixel_format) {
    if (!cam || !dev_path) return -1;
    memset(cam, 0, sizeof(V4L2Camera));
    strncpy(cam->dev_name, dev_path, sizeof(cam->dev_name) - 1);
    cam->width = width;
    cam->height = height;
    cam->pixel_format = pixel_format;
    cam->fd = -1;

    cam->fd = open(dev_path, O_RDWR | O_NONBLOCK, 0);
    if (cam->fd < 0) {
        fprintf(stderr, "[-] Cannot open '%s': %d, %s\n", dev_path, errno, strerror(errno));
        return -1;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixel_format;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (-1 == xioctl(cam->fd, VIDIOC_S_FMT, &fmt)) {
        fprintf(stderr, "[-] VIDIOC_S_FMT failed on '%s': %s\n", dev_path, strerror(errno));
        close(cam->fd);
        cam->fd = -1;
        return -1;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = V4L2_MAX_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(cam->fd, VIDIOC_REQBUFS, &req)) {
        fprintf(stderr, "[-] VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        close(cam->fd);
        cam->fd = -1;
        return -1;
    }

    cam->buffer_count = req.count;
    for (int i = 0; i < (int)req.count; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (-1 == xioctl(cam->fd, VIDIOC_QUERYBUF, &buf)) {
            fprintf(stderr, "[-] VIDIOC_QUERYBUF failed: %s\n", strerror(errno));
            close(cam->fd);
            cam->fd = -1;
            return -1;
        }

        cam->buffers[i].length = buf.length;
        cam->buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam->fd, buf.m.offset);
        if (MAP_FAILED == cam->buffers[i].start) {
            fprintf(stderr, "[-] mmap failed: %s\n", strerror(errno));
            close(cam->fd);
            cam->fd = -1;
            return -1;
        }
    }

    return 0;
}

int v4l2_camera_start(V4L2Camera* cam) {
    if (!cam || cam->fd < 0) return -1;

    for (int i = 0; i < cam->buffer_count; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (-1 == xioctl(cam->fd, VIDIOC_QBUF, &buf)) {
            return -1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(cam->fd, VIDIOC_STREAMON, &type)) {
        return -1;
    }

    cam->is_streaming = true;
    return 0;
}

int v4l2_camera_get_frame(V4L2Camera* cam, const void** frame_data, size_t* frame_size, int* buf_index) {
    if (!cam || !cam->is_streaming) return -1;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(cam->fd, VIDIOC_DQBUF, &buf)) {
        return -1;
    }

    *frame_data = cam->buffers[buf.index].start;
    *frame_size = buf.bytesused;
    *buf_index = buf.index;
    return 0;
}

int v4l2_camera_release_frame(V4L2Camera* cam, int buf_index) {
    if (!cam || cam->fd < 0 || buf_index < 0 || buf_index >= cam->buffer_count) return -1;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = buf_index;

    return xioctl(cam->fd, VIDIOC_QBUF, &buf);
}

void v4l2_camera_close(V4L2Camera* cam) {
    if (!cam || cam->fd < 0) return;

    if (cam->is_streaming) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(cam->fd, VIDIOC_STREAMOFF, &type);
        cam->is_streaming = false;
    }

    for (int i = 0; i < cam->buffer_count; ++i) {
        if (cam->buffers[i].start && cam->buffers[i].start != MAP_FAILED) {
            munmap(cam->buffers[i].start, cam->buffers[i].length);
        }
    }

    close(cam->fd);
    cam->fd = -1;
}
