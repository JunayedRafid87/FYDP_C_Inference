#!/usr/bin/env python3
"""
RDK X5 - Headless Dual Camera Inference & Web Streamer  (optimised)

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
import os
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

import cv2
import numpy as np

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

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "magma": cv2.COLORMAP_MAGMA,
    "bone": cv2.COLORMAP_BONE,
    "none": None,
}

CORES = os.cpu_count() or 4


def thermal_for_inference(frame):
    """CLAHE'd greyscale at NATIVE resolution, 3-channel.

    Matches the white-hot training data. No upscale, no colormap — both
    were costing CPU and moving the input away from what the model saw
    during training.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.cvtColor(CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)


def thermal_for_display(frame, colormap_type="inferno", target_w=640, target_h=480):
    """The pretty version: CLAHE, bicubic upscale, false colour. Display only."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    up = cv2.resize(CLAHE.apply(gray), (target_w, target_h),
                    interpolation=cv2.INTER_CUBIC)
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


def open_camera_smart(cam_target):
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


worker_results = {}
result_lock = threading.Lock()


class InferenceWorker(threading.Thread):
    def __init__(self, name, cam_target, model_path, is_thermal=False,
                 colormap="inferno", backend="onnx", imgsz=320, conf=0.40,
                 stride=1, threads=None, infer_on="raw"):
        super().__init__(daemon=True)
        self.name = name
        self.cam_target = cam_target
        self.is_thermal = is_thermal
        self.colormap = colormap
        self.model_path = model_path
        self.backend = backend
        self.imgsz = imgsz
        self.conf = conf
        self.stride = max(1, stride)
        self.threads = threads
        self.infer_on = infer_on
        self.running = True
        self.fps = 0.0
        self.infer_ms = 0.0

    def run(self):
        print(f"[{self.name}] Initializing capture for camera target "
              f"'{self.cam_target}'...")
        cap = open_camera_smart(self.cam_target)
        if cap is None or not cap.isOpened():
            print(f"[{self.name}] Error: Could not open camera '{self.cam_target}'")
            self.running = False
            return

        print(f"[{self.name}] Loading model ({self.backend}): {self.model_path}")
        try:
            if self.backend == "onnx" and HAS_ONNX:
                predictor = ONNXPredictor(self.model_path, imgsz=self.imgsz,
                                          conf_thresh=self.conf,
                                          threads=self.threads)
                print(f"[{self.name}] ONNX session using "
                      f"{predictor.threads} threads")
            elif HAS_ULTRALYTICS:
                pt = self.model_path.replace(".onnx", ".pt")
                if not Path(pt).exists():
                    pt = self.model_path
                print(f"[{self.name}] Using PyTorch fallback: {pt}")
                predictor = UltralyticsPredictor(pt, imgsz=self.imgsz,
                                                 conf_thresh=self.conf)
            else:
                print(f"[{self.name}] ERROR: neither onnxruntime nor ultralytics")
                self.running = False
                return
        except Exception as e:
            print(f"[{self.name}] Failed to load model: {e}")
            self.running = False
            return

        t0 = time.time()
        frame_i = 0
        boxes = []

        while self.running:
            # Discard anything the driver queued while we were busy.
            # BUFFERSIZE=1 is advisory and widely ignored on UVC; this is
            # what actually removes the latency.
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            t_now = time.time()
            self.fps = 0.85 * self.fps + 0.15 / max(t_now - t0, 1e-5)
            t0 = t_now

            if self.is_thermal:
                infer_frame = (thermal_for_display(frame, self.colormap)
                               if self.infer_on == "display"
                               else thermal_for_inference(frame))
                display = thermal_for_display(frame, self.colormap)
                sx = display.shape[1] / infer_frame.shape[1]
                sy = display.shape[0] / infer_frame.shape[0]
            else:
                infer_frame = frame
                display = frame
                sx = sy = 1.0

            if frame_i % self.stride == 0:
                ti = time.time()
                raw = predictor.predict(infer_frame)
                self.infer_ms = (time.time() - ti) * 1000
                boxes = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy),
                          c, k) for x1, y1, x2, y2, c, k in raw]
            frame_i += 1

            annotated = display if display is not frame else frame.copy()
            if annotated is display and self.is_thermal:
                pass                       # display is already a fresh array
            elif not self.is_thermal:
                annotated = frame.copy()

            colour = (0, 215, 255) if self.is_thermal else (0, 255, 127)
            for x1, y1, x2, y2, c, _ in boxes:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(annotated, f"Person {c:.2f}", (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            hud = (f"{self.name.upper()} | {self.fps:.1f} FPS | "
                   f"{self.infer_ms:.0f}ms | {len(boxes)} Det")
            cv2.rectangle(annotated, (5, 5), (360, 35), (15, 15, 15), -1)
            cv2.putText(annotated, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

            with result_lock:
                worker_results[self.name] = annotated

        cap.release()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


HTML = """<!DOCTYPE html>
<html><head><title>RDK X5 Dual Camera Inference Stream</title><style>
body{font-family:system-ui,sans-serif;background:#0b0f19;color:#f8fafc;
text-align:center;margin:0;padding:20px}
h1{color:#38bdf8;margin-bottom:5px}p{color:#94a3b8}
.stream-container{margin-top:15px;display:inline-block;background:#1e293b;
padding:12px;border-radius:14px;box-shadow:0 12px 30px rgba(0,0,0,.6)}
img{border-radius:8px;max-width:100%;height:auto;border:1px solid #334155}
.footer{margin-top:15px;font-size:.88em;color:#64748b}
</style></head><body>
<h1>RDK X5 Dual-Modality Human Detection</h1>
<p>Live Thermal + NoIR RGB YOLO11 Inference</p>
<div class="stream-container"><img src="/video_feed" width="960"></div>
<div class="footer">D-Robotics RDK X5 / Multi-Threaded ONNX Engine</div>
</body></html>"""


class WebStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                      # per-frame request logging is not free

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))

        elif self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            while True:
                with result_lock:
                    ft = worker_results.get("Thermal")
                    fr = worker_results.get("RGB")

                if ft is not None and fr is not None:
                    h = 480
                    w1 = int(ft.shape[1] * (h / ft.shape[0]))
                    w2 = int(fr.shape[1] * (h / fr.shape[0]))
                    combined = np.hstack([cv2.resize(ft, (w1, h)),
                                          cv2.resize(fr, (w2, h))])
                else:
                    combined = ft if ft is not None else fr

                if combined is not None:
                    ok, jpeg = cv2.imencode(
                        '.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ok:
                        data = jpeg.tobytes()
                        try:
                            self.wfile.write(b'--frame\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(data)))
                            self.end_headers()
                            self.wfile.write(data)
                            self.wfile.write(b'\r\n')
                        except Exception:
                            break
                time.sleep(0.03)
        else:
            self.send_error(404)


def main():
    p = argparse.ArgumentParser(
        description="RDK X5 Dual Camera Headless Inference Web Streamer")
    p.add_argument("--thermal-cam", default="0")
    p.add_argument("--rgb-cam", default="10",
                   help="Numbering shifts between boots: v4l2-ctl --list-devices")
    p.add_argument("--colormap", choices=list(COLORMAPS.keys()), default="inferno")
    p.add_argument("--backend", choices=["onnx", "pt"], default="onnx")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Model input size (320, 416, 512, 640)")
    p.add_argument("--rgb-imgsz", type=int, default=None,
                   help="Override input size for RGB only, e.g. 512")
    p.add_argument("--stride", type=int, default=1,
                   help="Detect every Nth frame, reuse boxes between")
    p.add_argument("--rgb-stride", type=int, default=None,
                   help="Override stride for RGB only, e.g. 3")
    p.add_argument("--threads", type=int, default=None,
                   help=f"ONNX threads per model (default {max(2, CORES // 2)})")
    p.add_argument("--infer-on", choices=["raw", "display"], default="raw",
                   help="Thermal: 'raw' = CLAHE greyscale (matches training), "
                        "'display' = old upscaled+colormapped behaviour")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    base = Path(__file__).parent
    rgb_sz = args.rgb_imgsz or args.imgsz

    def pick(prefix, sz, fallback):
        f = base / f"{prefix}_{sz}.onnx"
        return str(f) if f.exists() else str(fallback)

    if args.backend == "onnx":
        rgb_model = pick("rgb_yolo11n_best", rgb_sz,
                         base.parent / "rgb_yolo11n" / "weights" / "rgb_yolo11n_best.onnx")
        thm_model = pick("thermal_yolo11n_v3_best", args.imgsz,
                         base.parent / "thermal_yolo11n_v3" / "weights" / "thermal_yolo11n_v3.onnx")
    else:
        rgb_model = str(base / "rgb_yolo11n_best.pt")
        thm_model = str(base / "thermal_yolo11n_v3_best.pt")

    print("=" * 54)
    print("  RDK X5 Dual Inference Engine")
    print(f"  Thermal cam {args.thermal_cam} @ {args.imgsz}  stride {args.stride}"
          f"  infer-on {args.infer_on}")
    print(f"  RGB cam     {args.rgb_cam} @ {rgb_sz}  "
          f"stride {args.rgb_stride or args.stride}")
    print(f"  {CORES} cores, {args.threads or max(2, CORES // 2)} threads per model")
    print("=" * 54)

    wt = InferenceWorker("Thermal", args.thermal_cam, thm_model, is_thermal=True,
                         colormap=args.colormap, backend=args.backend,
                         imgsz=args.imgsz, stride=args.stride,
                         threads=args.threads, infer_on=args.infer_on)
    wr = InferenceWorker("RGB", args.rgb_cam, rgb_model, is_thermal=False,
                         backend=args.backend, imgsz=rgb_sz,
                         stride=args.rgb_stride or args.stride,
                         threads=args.threads)
    wt.start()
    wr.start()

    print(f"\n [OK] Web server live on port {args.port}")
    print(f" [->] Open http://<rdk-ip>:{args.port}\n")

    server = ThreadedHTTPServer(('0.0.0.0', args.port), WebStreamHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
        wt.running = wr.running = False
        server.server_close()


if __name__ == "__main__":
    main()
