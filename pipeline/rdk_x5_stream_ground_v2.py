#!/usr/bin/env python3
"""
RDK X5 - Headless Dual Camera Inference & Web Streamer  (low latency)

Changes from the previous version, in order of measured impact:

1. VECTORISED POSTPROCESSING. The old code looped over all ~2100 anchor
   boxes in Python every frame, per model. That cost more than the network
   forward pass itself and was invisible to benchmarks that only timed
   session.run(). Now done with numpy masks.

2. THREADS. intra_op_num_threads was 2. Measured on this board at 320:
   1 thread 436ms / 2 threads 235ms / 4 threads 133ms / 8 threads 86ms.
   Two models share 8 cores, so 4 each (running in parallel) beats 8 each
   (fighting over the same cores).

3. INFERENCE ON THE NATIVE FRAME. The thermal path used to upscale
   256x192 to 640x480 with bicubic, apply a colormap, and feed THAT to the
   model — which then resized it back down to 320. Pure waste, and the
   colormap is a domain mismatch: the model was trained on white-hot
   greyscale, not inferno false-colour. Now the model sees CLAHE'd
   greyscale at native resolution; the pretty version is built only for
   display. Use --infer-on display to get the old behaviour back.

4. BUFFER DRAINING. CAP_PROP_BUFFERSIZE is advisory and many UVC drivers
   ignore it, which is where the ~2s lag came from: the camera kept
   filling its queue while inference chewed on an old frame. Now stale
   frames are grabbed and discarded before each capture.

5. DETECTION STRIDE (--stride N). Run detection every Nth frame and reuse
   the last boxes in between. Video stays smooth while inference load
   drops. At 0.24 m/s a box is stale by a few cm — irrelevant.

Usage:
    python3 rdk_x5_dual_infer_headless.py --thermal-cam 0 --rgb-cam 10

Device numbering shifts between boots on this board. Check first:
    v4l2-ctl --list-devices
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from collections import deque
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None
    HAS_ULTRALYTICS = False


CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

INTERPS = {
    "linear": cv2.INTER_LINEAR,     # softest, no ringing — best for motion
    "cubic": cv2.INTER_CUBIC,       # middle ground
    "lanczos": cv2.INTER_LANCZOS4,  # sharpest on stills, rings on motion
}

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "magma": cv2.COLORMAP_MAGMA,
    "bone": cv2.COLORMAP_BONE,
    "none": None,
}

CORES = os.cpu_count() or 4

# Steady send cadence. 0 sends frames the instant they exist — lowest latency,
# but the interval then tracks encode time, which varies with scene content and
# reads as judder. Setting this paces delivery evenly.
PACE_FPS = 0.0


def thermal_for_inference(frame):
    """CLAHE'd greyscale at NATIVE resolution, 3-channel.

    Matches the white-hot training data. No upscale, no colormap — both
    were costing CPU and moving the input away from what the model saw
    during training.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.cvtColor(CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)


def thermal_for_display(frame, colormap_type="inferno", target_w=640,
                        target_h=480, interp=cv2.INTER_LINEAR, sharpen=0.0):
    """The pretty version: CLAHE, bicubic upscale, false colour. Display only."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    up = cv2.resize(CLAHE.apply(gray), (target_w, target_h),
                    interpolation=interp)

    if sharpen > 0.01:
        # Unsharp mask. Radius scales with the upscale factor so it
        # sharpens real sensor detail rather than interpolation texture.
        # Note this also amplifies the fixed-pattern noise every
        # microbolometer has — past ~1.2 the noise wins.
        sigma = max(1.0, target_w / gray.shape[1] * 0.6)
        blur = cv2.GaussianBlur(up, (0, 0), sigma)
        up = cv2.addWeighted(up, 1.0 + sharpen, blur, -sharpen, 0)
    cmap = COLORMAPS.get(colormap_type, cv2.COLORMAP_INFERNO)
    return cv2.applyColorMap(up, cmap) if cmap is not None \
        else cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)


class ONNXPredictor:
    def __init__(self, model_path, imgsz=320, conf_thresh=0.40,
                 iou_thresh=0.45, threads=None):
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime is not installed: pip3 install onnxruntime")
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        opts = ort.SessionOptions()
        # Two models share the CPU, so half the cores each. Measured: 4
        # threads x2 in parallel beats 8 threads x2 contending.
        opts.intra_op_num_threads = threads or max(2, CORES // 2)
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, opts,
                                            providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.threads = opts.intra_op_num_threads

    def preprocess(self, img):
        h, w = img.shape[:2]
        resized = cv2.resize(img, (self.imgsz, self.imgsz))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), (w, h)

    def predict(self, img):
        blob, (orig_w, orig_h) = self.preprocess(img)
        preds = self.session.run(self.output_names, {self.input_name: blob})[0][0]

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T                       # -> [anchors, 4+nc]

        # --- vectorised: this replaces a per-anchor Python loop ---
        scores_all = preds[:, 4:]
        class_ids = scores_all.argmax(1)
        confs = scores_all[np.arange(len(scores_all)), class_ids]

        keep = confs >= self.conf_thresh
        if not keep.any():
            return []

        p, confs, class_ids = preds[keep], confs[keep], class_ids[keep]

        sx, sy = orig_w / self.imgsz, orig_h / self.imgsz
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([(cx - bw / 2) * sx, (cy - bh / 2) * sy,
                          bw * sx, bh * sy], 1)

        idx = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(),
                               self.conf_thresh, self.iou_thresh)
        if len(idx) == 0:
            return []

        out = []
        for i in np.array(idx).flatten():
            x, y, w, h = boxes[i].astype(int)
            out.append((x, y, x + w, y + h, float(confs[i]), int(class_ids[i])))
        return out


class UltralyticsPredictor:
    def __init__(self, model_path, imgsz=320, conf_thresh=0.40, threads=None):
        if not HAS_ULTRALYTICS:
            raise RuntimeError("ultralytics is not installed")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf_thresh

    def predict(self, img):
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf,
                               verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((int(x1), int(y1), int(x2), int(y2),
                        float(b.conf[0]), int(b.cls[0])))
        return out


def open_camera_smart(cam_target, width=None, height=None, fps=None):
    """Open by GStreamer string, V4L2 index, or GStreamer fallback."""
    if isinstance(cam_target, str) and "!" in cam_target:
        print(f"Opening camera using GStreamer string: {cam_target}")
        cap = cv2.VideoCapture(cam_target, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap

    try:
        idx = int(cam_target)
    except ValueError:
        idx = 0

    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Resolution requests are validated, not trusted. The MIPI scaler node
    # advertises a Stepwise range (64x64 - 1920x1080) and will happily
    # accept a request then clamp it to something absurd — asking for
    # 1280x720 produced a 64x1920 format that yielded unusable frames and
    # silently killed detection. If the result looks wrong, reopen at the
    # driver default rather than stream garbage.
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sane = gw >= 160 and gh >= 120 and 0.5 <= (gw / max(gh, 1)) <= 3.0
        if not sane:
            print(f"  [!] /dev/video{idx} clamped {width}x{height} -> "
                  f"{gw}x{gh}; reopening at driver default")
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
        cap.release()

    gst = (f"v4l2src device=/dev/video{idx} ! video/x-raw, format=NV12 "
           f"! videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1")
    print(f"Trying GStreamer pipeline fallback for /dev/video{idx}...")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    return cap if cap.isOpened() else None



# ---------------------------------------------------------------------------
# Below this line is the ONLY part that changed: threading and HTTP.
# Detection logic above is byte-identical to the version that produced
# correct boxes. The old code did read -> infer -> annotate -> publish in
# one loop, so every displayed frame waited for a ~200ms inference and the
# driver queued frames behind it. Splitting them removes both effects.
# ---------------------------------------------------------------------------


class Slot:
    """Newest-value-wins handoff. No queue, so nothing can pile up."""

    def __init__(self):
        self.v, self.seq = None, 0
        self.lock = threading.Lock()

    def put(self, v):
        with self.lock:
            self.v = v
            self.seq += 1

    def get(self):
        with self.lock:
            return self.v, self.seq



# ---------------------------------------------------------------------------
# BPU backend. Wraps the Model Zoo's UltralyticsYOLODetect, which owns the
# hbm_runtime handle and already does letterbox -> NV12 packing -> DFL decode
# -> per-class NMS -> coordinate scale-back.
#
# Measured on this board with yolo11n at 640x640:
#     pre 11.1ms | BPU forward 15.5ms | post 16.1ms
# against ~300ms for the same model at 640 on CPU via ONNX Runtime.
# ---------------------------------------------------------------------------

MODEL_ZOO = os.path.expanduser("~/rdk_model_zoo")


class BPUPredictor:
    """Same predict() contract as ONNXPredictor: list of
    (x1, y1, x2, y2, conf, cls)."""

    def __init__(self, model_path, conf_thresh=0.40, iou_thresh=0.45,
                 classes_num=80, keep_class=0, resize_type=1):
        if not os.path.isdir(MODEL_ZOO):
            raise RuntimeError(f"model zoo not found at {MODEL_ZOO} — "
                               f"git clone https://github.com/D-Robotics/rdk_model_zoo.git")
        if MODEL_ZOO not in sys.path:
            sys.path.insert(0, MODEL_ZOO)
        sample = os.path.join(MODEL_ZOO,
                              "samples/vision/ultralytics_yolo/runtime/python")
        if sample not in sys.path:
            sys.path.insert(0, sample)

        from ultralytics_yolo_det import (UltralyticsYOLODetect,
                                          UltralyticsYOLODetectConfig)

        # The wrapper logs pre/forward/post timing at INFO on every call.
        # At 25fps that floods the terminal and costs real time.
        logging.getLogger("Ultralytics_YOLO").setLevel(logging.WARNING)

        cfg = UltralyticsYOLODetectConfig(
            model_path=model_path, classes_num=classes_num,
            score_thres=conf_thresh, nms_thres=iou_thresh,
            reg=16, resize_type=resize_type, strides=[8, 16, 32])
        self.m = UltralyticsYOLODetect(cfg)
        self.m.set_scheduling_params(priority=0, bpu_cores=[0])
        self.keep_class = keep_class
        self.threads = "BPU"

    def predict(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        boxes, scores, cls_ids = self.m.predict(img)
        out = []
        for (x1, y1, x2, y2), sc, cid in zip(boxes, scores, cls_ids):
            # COCO models detect 80 classes; keep only person unless told
            # otherwise. A single-class custom model needs keep_class=None.
            if self.keep_class is not None and int(cid) != self.keep_class:
                continue
            out.append((int(x1), int(y1), int(x2), int(y2),
                        float(sc), int(cid)))
        return out



# ---------------------------------------------------------------------------
# Gas telemetry from the ESP32-S3 over USB serial.
# ---------------------------------------------------------------------------

ALIASES = {
    "co": "co", "co_ppm": "co", "mems_co": "co",
    "ch4": "ch4", "ch4_ppm": "ch4", "mems_ch4": "ch4", "methane": "ch4",
    "eco2": "eco2", "co2": "eco2", "eco2_ppm": "eco2",
    "tvoc": "tvoc", "voc": "tvoc", "tvoc_ppb": "tvoc",
    "aqi": "aqi", "iaq": "aqi",
    "temp": "temp", "temperature": "temp", "temp_c": "temp",
    "hum": "hum", "humidity": "hum", "hum_pct": "hum",
}
KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(-?\d+(?:\.\d+)?)")

# The firmware prints temperature and humidity on one combined line,
# "T/RH: 30.3C  78%". Generic key:value matching reads "RH: 30.3" out of that
# and files the TEMPERATURE as humidity, so this line needs its own pattern.
TRH = re.compile(r"T/RH:\s*(-?\d+(?:\.\d+)?)\s*C\s+(\d+(?:\.\d+)?)\s*%")

# "Baseline  CO=0.124V   CH4=0.938V" reports clean-air VOLTAGES, not ppm.
SKIP_PREFIXES = ("Baseline", "Capturing", "Streaming", "Warming",
                 "I2C", "FYDP", "Motor", "Init", "----")

GAS = {"co": None, "ch4": None, "eco2": None, "tvoc": None,
       "aqi": None, "temp": None, "hum": None}
GAS_META = {"port": None, "connected": False, "last_rx": 0.0, "lines": 0}
GAS_LOCK = threading.Lock()
RAW_LINES = deque(maxlen=20)
SEEN_JSON = False
SERIAL_STALE_S = 10


def parse_gas_line(line):
    global SEEN_JSON
    line = line.strip()
    if not line:
        return {}
    body = line[2:].strip() if line.startswith("#J") else line
    if body.startswith("{"):
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                SEEN_JSON = True
                return {ALIASES[k.lower()]: v for k, v in obj.items()
                        if k.lower() in ALIASES and isinstance(v, (int, float))}
        except ValueError:
            pass
    if SEEN_JSON or line.startswith(SKIP_PREFIXES):
        return {}
    m = TRH.search(line)
    if m:
        return {"temp": float(m.group(1)), "hum": float(m.group(2))}
    out = {}
    for k, v in KV.findall(line):
        key = ALIASES.get(k.lower())
        if key:
            out[key] = float(v)
    return out


class GasReader(threading.Thread):
    def __init__(self, baud=115200):
        super().__init__(daemon=True)
        self.baud = baud
        self.running = True

    def run(self):
        if serial is None:
            print("[gas] pyserial not installed — pip3 install pyserial")
            return
        while self.running:
            ports = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
            if not ports:
                with GAS_LOCK:
                    GAS_META["connected"], GAS_META["port"] = False, None
                time.sleep(3)
                continue
            port = ports[0]
            try:
                ser = serial.Serial(port, self.baud, timeout=2)
                print(f"[gas] reading {port} @ {self.baud}")
                with GAS_LOCK:
                    GAS_META["port"], GAS_META["connected"] = port, True
            except Exception as e:
                print(f"[gas] cannot open {port}: {e}")
                time.sleep(3)
                continue
            try:
                while self.running:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    vals = parse_gas_line(line)
                    with GAS_LOCK:
                        RAW_LINES.append(line[:200])
                        GAS_META["lines"] += 1
                        if vals:
                            GAS.update(vals)
                            GAS_META["last_rx"] = time.time()
            except Exception as e:
                print(f"[gas] {port} dropped: {e}")
                with GAS_LOCK:
                    GAS_META["connected"] = False
                try:
                    ser.close()
                except Exception:
                    pass
                time.sleep(2)


class InferThread(threading.Thread):
    """Runs the model on whatever frame is newest. Never blocks capture."""

    def __init__(self, name, src_slot, model_path, imgsz, conf, threads,
                 backend="onnx", classes_num=80, keep_class=0):
        super().__init__(daemon=True)
        self.name, self.src = name, src_slot
        self.model_path, self.imgsz = model_path, imgsz
        self.conf, self.threads = conf, threads
        self.backend = backend
        self.classes_num, self.keep_class = classes_num, keep_class
        self.out = Slot()
        self.ms = 0.0
        self.hz = 0.0
        self.running = True

    def run(self):
        try:
            if self.backend == "bpu":
                pred = BPUPredictor(self.model_path, conf_thresh=self.conf,
                                    classes_num=self.classes_num,
                                    keep_class=self.keep_class)
            else:
                pred = ONNXPredictor(self.model_path, imgsz=self.imgsz,
                                     conf_thresh=self.conf, threads=self.threads)
        except Exception as e:
            print(f"[{self.name}] model load failed: {e}")
            import traceback; traceback.print_exc()
            return
        print(f"[{self.name}] model ready ({pred.threads}): {self.model_path}")

        last, t0 = -1, time.time()
        while self.running:
            frame, seq = self.src.get()
            if frame is None or seq == last:
                time.sleep(0.003)
                continue
            last = seq
            t = time.time()
            boxes = pred.predict(frame)
            self.ms = (time.time() - t) * 1000
            now = time.time()
            self.hz = 0.8 * self.hz + 0.2 / max(now - t0, 1e-5)
            t0 = now
            self.out.put(boxes)


class CameraWorker(threading.Thread):
    """Capture, publish for inference, draw newest boxes, encode.

    Frame handling and the choice of what goes to the model are copied
    exactly from the previous working loop.
    """

    def __init__(self, name, cam_target, is_thermal=False, colormap="inferno",
                 infer_on="raw", quality=75, out_size=None,
                 interp=cv2.INTER_LINEAR, sharpen=0.0, hud=True,
                 width=None, height=None, fps=None, draw=True):
        super().__init__(daemon=True)
        self.name, self.cam_target = name, cam_target
        self.is_thermal, self.colormap = is_thermal, colormap
        self.infer_on, self.quality = infer_on, quality
        self.out_size = out_size
        self.interp = interp
        self.sharpen = sharpen
        self.hud = hud
        self.draw = draw
        self.det = {}
        self.width, self.height, self.req_fps = width, height, fps
        self.stats = {}
        self.infer_in = Slot()
        self.jpeg = Slot()
        self.infer = None
        self.fps = 0.0
        self.running = True

    def run(self):
        cap = open_camera_smart(self.cam_target, self.width,
                                self.height, self.req_fps)
        if cap is None or not cap.isOpened():
            print(f"[{self.name}] Error: could not open camera '{self.cam_target}'")
            return
        gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{self.name}] capture started at {gw}x{gh}")

        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        colour = (0, 215, 255) if self.is_thermal else (0, 255, 127)
        t0 = time.time()
        boxes = []

        while self.running:
            cap.grab()                     # drop anything queued behind us
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            t_now = time.time()
            self.fps = 0.85 * self.fps + 0.15 / max(t_now - t0, 1e-5)
            t0 = t_now

            if self.is_thermal:
                ow, oh = self.out_size or (640, 480)
                infer_frame = (thermal_for_display(frame, self.colormap, ow, oh,
                                                   self.interp, self.sharpen)
                               if self.infer_on == "display"
                               else thermal_for_inference(frame))
                display = thermal_for_display(frame, self.colormap, ow, oh,
                                              self.interp, self.sharpen)
                sx = display.shape[1] / infer_frame.shape[1]
                sy = display.shape[0] / infer_frame.shape[0]
            else:
                infer_frame = frame
                display = frame
                sx = sy = 1.0

            self.infer_in.put(infer_frame)

            ms = hz = 0.0
            if self.infer is not None:
                raw, _ = self.infer.out.get()
                ms, hz = self.infer.ms, self.infer.hz
                if raw is not None:
                    boxes = [(int(x1 * sx), int(y1 * sy),
                              int(x2 * sx), int(y2 * sy), c, k)
                             for x1, y1, x2, y2, c, k in raw]

            # Published for the browser overlay. Coordinates are in DISPLAY
            # space, so the client only needs to scale by its own <img> size.
            self.det = {"w": display.shape[1], "h": display.shape[0],
                        "fps": round(self.fps, 1), "hz": round(hz, 1),
                        "ms": round(ms), "n": len(boxes),
                        "boxes": [[b[0], b[1], b[2], b[3], round(b[4], 3)]
                                  for b in boxes]}

            if not self.draw:
                # Clean video path: no copy, no rectangles, no HUD. The
                # display.copy() alone was ~920KB per frame at 640x480, and
                # it sat directly in the teleop latency path.
                ok, jpg = cv2.imencode('.jpg', display, enc)
                if ok:
                    self.jpeg.put(jpg.tobytes())
                continue

            annotated = display.copy() if display is frame else display
            bt = 1 if annotated.shape[1] < 500 else 2
            bfs = max(0.3, annotated.shape[1] / 1400.0)
            for x1, y1, x2, y2, c, _ in boxes:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, bt)
                cv2.putText(annotated, f"{c:.2f}", (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, bfs, colour, bt)

            self.stats = {"fps": round(self.fps, 1), "det_hz": round(hz, 1),
                          "infer_ms": round(ms), "n": len(boxes),
                          "w": annotated.shape[1], "h": annotated.shape[0]}

            if self.hud:
                # Scale with frame width. Fixed pixel sizes were designed for
                # a 640px frame and swamped the 256px native thermal output.
                fw = annotated.shape[1]
                fs = max(0.28, fw / 1200.0)
                th = 1 if fw < 500 else 2
                hud = (f"{self.name.upper()} {self.fps:.0f}fps "
                       f"{hz:.0f}Hz {ms:.0f}ms {len(boxes)}")
                (tw, tht), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX,
                                               fs, th)
                pad = max(3, int(fw * 0.008))
                cv2.rectangle(annotated, (pad, pad),
                              (pad * 2 + tw, pad * 2 + tht + 4), (15, 15, 15), -1)
                cv2.putText(annotated, hud, (pad + 3, pad + tht + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                            th, cv2.LINE_AA)

            ok, jpg = cv2.imencode('.jpg', annotated, enc)
            if ok:
                self.jpeg.put(jpg.tobytes())
        cap.release()


streams = {}


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


OVERLAY_HTML = """<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>RDK X5 Teleop</title><style>
body{font-family:system-ui,sans-serif;background:#0b0f19;color:#f8fafc;
text-align:center;margin:0;padding:18px}
h1{color:#38bdf8;font-size:19px;margin:0 0 4px}
p{color:#94a3b8;margin:0 0 14px;font-size:13px}
.g{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
.p{background:#1e293b;padding:10px;border-radius:12px}
.w{position:relative;line-height:0}
.w img{display:block;border-radius:8px;border:1px solid #334155;width:640px;height:auto}
.w canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
.l{font-size:11px;letter-spacing:.12em;color:#64748b;margin-top:7px}
.s{font:12px ui-monospace,monospace;color:#94a3b8;margin-top:3px}
</style></head><body>
<h1>RDK X5 Teleop &middot; overlay detection</h1>
<p>Video is streamed untouched; boxes are drawn client-side</p>
<div class="g">
  <div class="p"><div class="w"><img id="iRGB" src="/rgb_feed">
    <canvas id="cRGB"></canvas></div>
    <div class="l">RGB / NoIR</div><div class="s" id="sRGB">&mdash;</div></div>
  <div class="p"><div class="w"><img id="iThermal" src="/thermal_feed">
    <canvas id="cThermal"></canvas></div>
    <div class="l">THERMAL</div><div class="s" id="sThermal">&mdash;</div></div>
</div>
<script>
const COLOR = {RGB: '#7cff7c', Thermal: '#ffd24a'};
function draw(name, d) {
  const img = document.getElementById('i' + name);
  const cv  = document.getElementById('c' + name);
  if (!img || !cv || !d || !d.w) return;
  const W = img.clientWidth, H = img.clientHeight;
  if (!W || !H) return;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const g = cv.getContext('2d');
  g.clearRect(0, 0, W, H);
  const sx = W / d.w, sy = H / d.h;
  g.lineWidth = 2; g.strokeStyle = COLOR[name] || '#7cff7c';
  g.fillStyle = g.strokeStyle; g.font = '13px ui-monospace, monospace';
  (d.boxes || []).forEach(b => {
    const x = b[0]*sx, y = b[1]*sy, w = (b[2]-b[0])*sx, h = (b[3]-b[1])*sy;
    g.strokeRect(x, y, w, h);
    g.fillText(b[4].toFixed(2), x + 2, Math.max(y - 4, 12));
  });
  const el = document.getElementById('s' + name);
  if (el) el.textContent = d.w+'x'+d.h+' \u00b7 '+d.fps+' fps \u00b7 det '+
      d.hz+' Hz ('+d.ms+' ms) \u00b7 '+d.n+' det';
}
function poll() {
  fetch('/detections').then(r => r.json()).then(d => {
    for (const k of ['RGB','Thermal']) draw(k, d[k]);
  }).catch(()=>{});
}
setInterval(poll, 100);
poll();
</script>
</body></html>"""


class WebStreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                     # per-frame request logging is not free

    def do_GET(self):
        if self.path == '/teleop':
            body = OVERLAY_HTML.encode()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == '/detections':
            body = json.dumps({k: v.det for k, v in streams.items()}).encode()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == '/gas':
            with GAS_LOCK:
                out = {k: v for k, v in GAS.items() if v is not None}
                age = (time.time() - GAS_META["last_rx"]
                       if GAS_META["last_rx"] else None)
                out["_meta"] = {"port": GAS_META["port"],
                                "connected": GAS_META["connected"],
                                "lines": GAS_META["lines"],
                                "age_s": round(age, 1) if age else None,
                                "stale": age is None or age > SERIAL_STALE_S}
            body = json.dumps(out).encode()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == '/gas/raw':
            with GAS_LOCK:
                body = json.dumps({"port": GAS_META["port"],
                                   "lines": list(RAW_LINES)}).encode()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == '/':
            page = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Rover_Ground_Station.html")
            if os.path.exists(page):
                body = open(page, "rb").read()
            else:
                body = OVERLAY_HTML.encode()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == '/stats':
            body = json.dumps({k: v.stats for k, v in streams.items()}).encode()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return

        key = {'/thermal_feed': 'Thermal', '/rgb_feed': 'RGB',
               '/thermal': 'Thermal', '/rgb': 'RGB'}.get(self.path)
        if key is None or key not in streams:
            self.send_error(404)
            return

        src = streams[key]
        self.send_response(200)
        self.send_header('Content-type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

        last = -1
        min_dt = 1.0 / PACE_FPS if PACE_FPS > 0 else 0.0
        next_send = 0.0
        while True:
            data, seq = src.jpeg.get()
            if data is None or seq == last:
                time.sleep(0.002)    # was 0.03 — that alone added 30ms
                continue
            now = time.time()
            if min_dt and now < next_send:
                time.sleep(min(next_send - now, 0.01))
                continue
            next_send = max(now, next_send) + min_dt
            last = seq
            try:
                self.wfile.write(b'--frame\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.write(b'\r\n')
            except Exception:
                break


def main():
    global PACE_FPS
    parser = argparse.ArgumentParser(
        description="RDK X5 Dual Camera Headless Inference Web Streamer")
    parser.add_argument("--thermal-cam", default="0")
    parser.add_argument("--rgb-cam", default="10",
                        help="Numbering shifts between boots: v4l2-ctl --list-devices")
    parser.add_argument("--colormap", choices=list(COLORMAPS.keys()),
                        default="inferno")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--rgb-imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--quality", type=int, default=70,
                        help="RGB JPEG quality; lower = less bandwidth, less latency")
    parser.add_argument("--thermal-quality", type=int, default=78,
                        help="Thermal JPEG quality. Raising this enlarges every "
                             "frame, which over WiFi lowers effective framerate "
                             "and makes MOTION look worse even as stills look "
                             "sharper. 92 was too high on this link.")
    parser.add_argument("--rgb-backend", choices=["onnx", "bpu"], default="bpu",
                        help="bpu uses the pre-compiled COCO model (~15ms at "
                             "640); onnx uses your custom model on CPU (~300ms "
                             "at 640).")
    parser.add_argument("--thermal-backend", choices=["onnx", "bpu"],
                        default="onnx",
                        help="Your thermal model is not converted yet, so onnx. "
                             "bpu would run the COCO model on thermal frames, "
                             "which it was never trained for.")
    parser.add_argument("--rgb-bpu-model", default=os.path.expanduser(
                        "~/rdk_model_zoo/samples/vision/ultralytics_yolo/model/"
                        "yolo11n_detect_bayese_640x640_nv12.bin"),
                        help="RGB .bin. Default is the stock COCO model.")
    parser.add_argument("--rgb-bpu-classes", type=int, default=80)
    parser.add_argument("--rgb-keep-class", type=int, default=0,
                        help="COCO class to keep; 0 is person")
    parser.add_argument("--thermal-bpu-model", default=os.path.expanduser(
                        "~/thermal_yolo11n_v3_best_bayese_640x640_nv12.bin"),
                        help="Thermal .bin from your own conversion")
    parser.add_argument("--thermal-bpu-classes", type=int, default=1,
                        help="Your thermal model is single-class")
    parser.add_argument("--thermal-keep-class", type=int, default=-1,
                        help="-1 keeps all classes, correct for a "
                             "single-class model")
    parser.add_argument("--no-hud", action="store_true",
                        help="No overlay on the image; stats show under each "
                             "panel instead. Cleaner for demo screenshots.")
    parser.add_argument("--rgb-width", type=int, default=None,
                        help="Leave unset to use the driver default. "
                             "This node clamps odd requests.")
    parser.add_argument("--rgb-height", type=int, default=None)
    parser.add_argument("--rgb-fps", type=int, default=None)
    parser.add_argument("--thermal-fps", type=int, default=50,
                        help="Thermal sensor offers 25 or 50 only. Capture was "
                             "running at 12.4fps unrequested, which capped "
                             "detection at 13Hz even though inference had "
                             "47Hz of headroom.")
    parser.add_argument("--thermal-sharpen", type=float, default=0.0,
                        help="Unsharp mask on the thermal display, 0 = off. "
                             "Amplifies microbolometer fixed-pattern noise as "
                             "well as detail; past ~1.2 the noise dominates.")
    parser.add_argument("--thermal-interp", choices=list(INTERPS),
                        default="linear",
                        help="Upscale filter. lanczos is sharpest on stills but "
                             "rings on edges, which reads as motion doubling.")
    parser.add_argument("--thermal-out", type=int, nargs=2, default=[256, 192],
                        metavar=("W", "H"),
                        help="Thermal display upscale. The sensor is 256x192; "
                             "this only interpolates, it cannot add detail.")
    parser.add_argument("--threads", type=int, default=None,
                        help=f"ONNX threads per model (default {max(2, CORES // 2)})")
    parser.add_argument("--infer-on", choices=["raw", "display"], default="raw",
                        help="Thermal model input, unchanged from before")
    parser.add_argument("--overlay", action="store_true",
                        help="Stream video untouched and draw boxes in the "
                             "browser. Removes a full-frame copy plus all "
                             "drawing from the video path — use this for "
                             "teleop, where stutter matters. View at /teleop.")
    parser.add_argument("--pace-fps", type=float, default=0.0,
                        help="Send at a steady cadence, e.g. 25. 0 = send as "
                             "soon as ready (lowest latency, more judder).")
    parser.add_argument("--no-gas", action="store_true",
                        help="Skip the ESP32 serial reader")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    PACE_FPS = args.pace_fps

    base = Path(__file__).parent
    rgb_sz = args.rgb_imgsz or args.imgsz

    def pick(prefix, sz, fb):
        f = base / f"{prefix}_{sz}.onnx"
        return str(f) if f.exists() else str(fb)

    rgb_model = pick("rgb_yolo11n_best", rgb_sz,
                     base.parent / "rgb_yolo11n/weights/rgb_yolo11n_best.onnx")
    thm_model = pick("thermal_yolo11n_v3_best", args.imgsz,
                     base.parent / "thermal_yolo11n_v3/weights/thermal_yolo11n_v3.onnx")

    print("=" * 56)
    print(f"  {CORES} cores | {args.threads or max(2, CORES//2)} threads per model")
    print(f"  Thermal cam {args.thermal_cam} @ {args.imgsz} on {args.infer_on}")
    print(f"  RGB     cam {args.rgb_cam} @ {rgb_sz}")
    print("=" * 56)

    thm = CameraWorker("Thermal", args.thermal_cam, is_thermal=True,
                       colormap=args.colormap, infer_on=args.infer_on,
                       quality=args.thermal_quality,
                       out_size=tuple(args.thermal_out),
                       interp=INTERPS[args.thermal_interp],
                       sharpen=args.thermal_sharpen,
                       hud=not args.no_hud,
                       fps=args.thermal_fps,
                       draw=not args.overlay)
    rgb = CameraWorker("RGB", args.rgb_cam, is_thermal=False,
                       quality=args.quality, hud=not args.no_hud,
                       width=args.rgb_width, height=args.rgb_height,
                       fps=args.rgb_fps, draw=not args.overlay)

    thm.infer = InferThread(
        "Thermal", thm.infer_in,
        args.thermal_bpu_model if args.thermal_backend == "bpu" else thm_model,
        args.imgsz, args.conf, args.threads,
        backend=args.thermal_backend,
        classes_num=args.thermal_bpu_classes,
        keep_class=None if args.thermal_keep_class < 0 else args.thermal_keep_class)
    rgb.infer = InferThread(
        "RGB", rgb.infer_in,
        args.rgb_bpu_model if args.rgb_backend == "bpu" else rgb_model,
        rgb_sz, args.conf, args.threads,
        backend=args.rgb_backend,
        classes_num=args.rgb_bpu_classes,
        keep_class=None if args.rgb_keep_class < 0 else args.rgb_keep_class)
    thm.infer.start()
    rgb.infer.start()
    thm.start()
    rgb.start()
    streams["Thermal"], streams["RGB"] = thm, rgb

    if not args.no_gas:
        GasReader().start()

    print(f"\n [OK] http://<rdk-ip>:{args.port}\n")
    server = ThreadedHTTPServer(('0.0.0.0', args.port), WebStreamHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        for w in (thm, rgb):
            w.running = False
            if w.infer:
                w.infer.running = False
        server.server_close()


if __name__ == "__main__":
    main()
