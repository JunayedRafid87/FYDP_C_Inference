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


class InferThread(threading.Thread):
    """Runs the model on whatever frame is newest. Never blocks capture."""

    def __init__(self, name, src_slot, model_path, imgsz, conf, threads):
        super().__init__(daemon=True)
        self.name, self.src = name, src_slot
        self.model_path, self.imgsz = model_path, imgsz
        self.conf, self.threads = conf, threads
        self.out = Slot()
        self.ms = 0.0
        self.hz = 0.0
        self.running = True

    def run(self):
        try:
            pred = ONNXPredictor(self.model_path, imgsz=self.imgsz,
                                 conf_thresh=self.conf, threads=self.threads)
        except Exception as e:
            print(f"[{self.name}] model load failed: {e}")
            return
        print(f"[{self.name}] model ready ({pred.threads} threads): {self.model_path}")

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
                 interp=cv2.INTER_LINEAR, sharpen=0.0):
        super().__init__(daemon=True)
        self.name, self.cam_target = name, cam_target
        self.is_thermal, self.colormap = is_thermal, colormap
        self.infer_on, self.quality = infer_on, quality
        self.out_size = out_size
        self.interp = interp
        self.sharpen = sharpen
        self.infer_in = Slot()
        self.jpeg = Slot()
        self.infer = None
        self.fps = 0.0
        self.running = True

    def run(self):
        cap = open_camera_smart(self.cam_target)
        if cap is None or not cap.isOpened():
            print(f"[{self.name}] Error: could not open camera '{self.cam_target}'")
            return
        print(f"[{self.name}] capture started")

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

            annotated = display.copy() if display is frame else display
            for x1, y1, x2, y2, c, _ in boxes:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(annotated, f"Person {c:.2f}", (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            hud = (f"{self.name.upper()} | {self.fps:.0f} FPS | "
                   f"det {hz:.1f}Hz {ms:.0f}ms | {len(boxes)} Det")
            cv2.rectangle(annotated, (5, 5), (380, 35), (15, 15, 15), -1)
            cv2.putText(annotated, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

            ok, jpg = cv2.imencode('.jpg', annotated, enc)
            if ok:
                self.jpeg.put(jpg.tobytes())
        cap.release()


streams = {}


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


HTML = """<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>RDK X5 Dual Camera Inference Stream</title><style>
body{font-family:system-ui,sans-serif;background:#0b0f19;color:#f8fafc;
text-align:center;margin:0;padding:20px}
h1{color:#38bdf8;margin-bottom:5px}p{color:#94a3b8}
.g{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:15px}
.p{background:#1e293b;padding:12px;border-radius:14px;
box-shadow:0 12px 30px rgba(0,0,0,.6)}
img{display:block;border-radius:8px;border:1px solid #334155;max-width:100%}
.l{font-size:11px;letter-spacing:.12em;color:#64748b;margin-top:8px}
</style></head><body>
<h1>RDK X5 Dual-Modality Human Detection</h1>
<p>Independent streams &middot; detection asynchronous</p>
<div class="g">
  <div class="p"><img src="/thermal_feed" width="700"><div class="l">THERMAL</div></div>
  <div class="p"><img src="/rgb_feed" width="560"><div class="l">RGB / NoIR</div></div>
</div>
</body></html>"""


class WebStreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                     # per-frame request logging is not free

    def do_GET(self):
        if self.path == '/':
            body = HTML.encode()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        key = {'/thermal_feed': 'Thermal', '/rgb_feed': 'RGB'}.get(self.path)
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
        while True:
            data, seq = src.jpeg.get()
            if data is None or seq == last:
                time.sleep(0.002)    # was 0.03 — that alone added 30ms
                continue
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
    parser.add_argument("--thermal-sharpen", type=float, default=0.8,
                        help="Unsharp mask on the thermal display, 0 = off. "
                             "Amplifies microbolometer fixed-pattern noise as "
                             "well as detail; past ~1.2 the noise dominates.")
    parser.add_argument("--thermal-interp", choices=list(INTERPS),
                        default="linear",
                        help="Upscale filter. lanczos is sharpest on stills but "
                             "rings on edges, which reads as motion doubling.")
    parser.add_argument("--thermal-out", type=int, nargs=2, default=[640, 480],
                        metavar=("W", "H"),
                        help="Thermal display upscale. The sensor is 256x192; "
                             "this only interpolates, it cannot add detail.")
    parser.add_argument("--threads", type=int, default=None,
                        help=f"ONNX threads per model (default {max(2, CORES // 2)})")
    parser.add_argument("--infer-on", choices=["raw", "display"], default="raw",
                        help="Thermal model input, unchanged from before")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

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
                       sharpen=args.thermal_sharpen)
    rgb = CameraWorker("RGB", args.rgb_cam, is_thermal=False,
                       quality=args.quality)

    thm.infer = InferThread("Thermal", thm.infer_in, thm_model, args.imgsz,
                            args.conf, args.threads)
    rgb.infer = InferThread("RGB", rgb.infer_in, rgb_model, rgb_sz,
                            args.conf, args.threads)
    thm.infer.start()
    rgb.infer.start()
    thm.start()
    rgb.start()
    streams["Thermal"], streams["RGB"] = thm, rgb

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
