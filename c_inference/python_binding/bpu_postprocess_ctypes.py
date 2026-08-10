#!/usr/bin/env python3
"""
bpu_postprocess_ctypes.py
Python Ctypes wrapper for native C accelerated YOLO11 DFL decoding and NMS.
Plugs directly into rdk_x5_stream_ground_v10.py to eliminate Python CPU bottleneck.
"""

import ctypes
import os
import numpy as np

# Structure definitions matching postprocess.h
class DetectionBox(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("score", ctypes.c_float),
        ("class_id", ctypes.c_int),
    ]

class DetectionResult(ctypes.Structure):
    _fields_ = [
        ("boxes", DetectionBox * 300),
        ("count", ctypes.c_int),
        ("inference_time_ms", ctypes.c_float),
        ("postprocess_time_ms", ctypes.c_float),
    ]

class FeatureScale(ctypes.Structure):
    _fields_ = [
        ("grid_h", ctypes.c_int),
        ("grid_w", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("cls_ptr", ctypes.POINTER(ctypes.c_float)),
        ("reg_ptr", ctypes.POINTER(ctypes.c_float)),
        ("num_classes", ctypes.c_int),
    ]

class CPostProcessor:
    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(lib_dir, "libpostprocess.so"),
                os.path.join(lib_dir, "../build/libpostprocess.so"),
                "/usr/local/lib/libpostprocess.so",
            ]
            for c in candidates:
                if os.path.exists(c):
                    lib_path = c
                    break
        
        self.lib = None
        if lib_path and os.path.exists(lib_path):
            try:
                self.lib = ctypes.CDLL(lib_path)
                self._setup_signatures()
                self.lib.init_postprocess_luts()
                print(f"[CPostProcessor] Loaded native acceleration from: {lib_path}")
            except Exception as e:
                print(f"[CPostProcessor] Failed to load {lib_path}: {e}. Falling back to Python.")
        else:
            print("[CPostProcessor] Native library not found. Falling back to pure Python postprocessing.")

    def _setup_signatures(self):
        self.lib.init_postprocess_luts.argtypes = []
        self.lib.init_postprocess_luts.restype = None

        self.lib.decode_yolo11_outputs.argtypes = [
            ctypes.POINTER(FeatureScale),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(DetectionResult)
        ]
        self.lib.decode_yolo11_outputs.restype = None

    def is_available(self):
        return self.lib is not None

    def postprocess(self, outputs, conf_thresh=0.25, iou_thresh=0.45, target_w=640, target_h=640):
        """
        Decodes 6 BPU/ONNX output tensors:
        outputs = [cls_80, reg_80, cls_40, reg_40, cls_20, reg_20]
        """
        if not self.is_available():
            return None

        scales = (FeatureScale * 3)()
        shapes = [(80, 80, 8), (40, 40, 16), (20, 20, 32)]

        # Keep contiguous references
        contiguous_arrays = []
        for i in range(3):
            cls_arr = np.ascontiguousarray(outputs[i * 2], dtype=np.float32)
            reg_arr = np.ascontiguousarray(outputs[i * 2 + 1], dtype=np.float32)
            contiguous_arrays.extend([cls_arr, reg_arr])

            h, w, stride = shapes[i]
            num_classes = cls_arr.shape[-1]

            scales[i].grid_h = h
            scales[i].grid_w = w
            scales[i].stride = stride
            scales[i].num_classes = num_classes
            scales[i].cls_ptr = cls_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            scales[i].reg_ptr = reg_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        result = DetectionResult()
        self.lib.decode_yolo11_outputs(
            scales,
            3,
            ctypes.c_float(conf_thresh),
            ctypes.c_float(iou_thresh),
            ctypes.c_int(target_w),
            ctypes.c_int(target_h),
            ctypes.byref(result)
        )

        boxes = []
        for i in range(result.count):
            b = result.boxes[i]
            boxes.append({
                "box": [float(b.x1), float(b.y1), float(b.x2), float(b.y2)],
                "score": float(b.score),
                "class_id": int(b.class_id)
            })

        return boxes
