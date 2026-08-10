#!/usr/bin/env python3
"""
RDK X5 - Teleop video + thermal detection.

RGB is a pure passthrough for driving the rover: capture -> JPEG -> socket,
no model in the path. Thermal is the only inference stream, and it gets the
whole CPU.

TWO INDEPENDENT ENDPOINTS
    /rgb_feed      RGB only, runs at sensor rate
    /thermal_feed  thermal + boxes, runs at its own rate

    The previous version composited both cameras into one JPEG, which
    locked the fast stream to the slow one. Separate endpoints let RGB run
    at 60fps regardless of what the model is doing.

WHAT TO EXPECT ON THIS BOARD
    RGB video        50-60fps at 640x480, ~6ms to encode
    Thermal video    25fps (sensor cap is 25 or 50)
    Thermal detect   ~11Hz at imgsz 320 with 8 threads
                     ~24Hz at imgsz 224
                     ~32Hz at imgsz 192  (below the 256px sensor width,
                                          so small-target range suffers)

    Measured pure inference at 320: 1thr 436ms / 2thr 235ms / 4thr 133ms /
    8thr 86ms. 30Hz needs 33ms, so 320 on CPU cannot reach it. Either drop
    input size or move the model to the BPU.

    Glass-to-browser latency lands around 80-120ms. Sub-50ms is not
    reachable with MJPEG in a browser — decode and display buffering alone
    exceed that. WebRTC or a native client would be needed.

Usage:
    python3 rdk_x5_teleop.py --thermal-cam 0 --rgb-cam 10
    python3 rdk_x5_teleop.py --thermal-cam 0 --rgb-cam 10 --imgsz 192
    python3 rdk_x5_teleop.py --rgb-cam 10 --no-thermal      # teleop only

Device numbering shifts between boots:  v4l2-ctl --list-devices
"""

import argparse
import os
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


CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO, "jet": cv2.COLORMAP_JET,
    "magma": cv2.COLORMAP_MAGMA, "bone": cv2.COLORMAP_BONE, "none": None,
}
CORES = os.cpu_count() or 4


class Slot:
    """Newest-value-wins handoff. No queue, so no backlog and no lag."""

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


def open_camera(target, width=None, height=None, fps=None):
    if isinstance(target, str) and "!" in target:
        cap = cv2.VideoCapture(target, cv2.CAP_GSTREAMER)
        return cap if cap.isOpened() else None
    try:
        idx = int(target)
    except ValueError:
        idx = 0

    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if cap.isOpened():
        ok, f = cap.read()
        if ok and f is not None:
            return cap
        cap.release()

    gst = (f"v4l2src device=/dev/video{idx} ! video/x-raw, format=NV12 "
           f"! videoconvert ! video/x-raw, format=BGR "
           f"! appsink drop=true max-buffers=1 sync=false")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    return cap if cap.isOpened() else None


class RGBStream(threading.Thread):
    """Pure teleop passthrough. Encodes straight to JPEG — no BGR copy, no
    model, nothing between the sensor and the socket."""

    def __init__(self, target, quality, width, height, fps):
        super().__init__(daemon=True)
        self.target, self.quality = target, quality
        self.width, self.height, self.req_fps = width, height, fps
        self.jpeg = Slot()
        self.fps = 0.0
        self.enc_ms = 0.0
        self.running = True

    def run(self):
        cap = open_camera(self.target, self.width, self.height, self.req_fps)
        if cap is None:
            print(f"[RGB] cannot open camera '{self.target}'")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[RGB] streaming {w}x{h} from '{self.target}'")

        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        t0 = time.time()
        while self.running:
            if not cap.grab():
                time.sleep(0.002)
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - t0, 1e-5)
            t0 = now

            te = time.time()
            ok, jpg = cv2.imencode('.jpg', frame, enc)
            self.enc_ms = (time.time() - te) * 1000
            if ok:
                self.jpeg.put(jpg.tobytes())
        cap.release()


class Predictor:
    def __init__(self, path, imgsz, conf=0.40, iou=0.45, threads=None):
        opts = ort.SessionOptions()
        # Only one model now, so give it every core.
        opts.intra_op_num_threads = threads or CORES
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(path, opts,
                                      providers=['CPUExecutionProvider'])
        self.inp = self.s.get_inputs()[0].name
        self.outs = [o.name for o in self.s.get_outputs()]
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.threads = opts.intra_op_num_threads

    def predict(self, gray):
        h, w = gray.shape[:2]
        r = cv2.resize(gray, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)
        blob = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0

        preds = self.s.run(self.outs, {self.inp: blob})[0][0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        # Vectorised. The original looped over ~2100 anchors in Python,
        # which cost more than the forward pass itself.
        sc = preds[:, 4:]
        cf = sc[np.arange(len(sc)), sc.argmax(1)]
        keep = cf >= self.conf
        if not keep.any():
            return []

        p, cf = preds[keep], cf[keep]
        sx, sy = w / self.imgsz, h / self.imgsz
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([(cx - bw / 2) * sx, (cy - bh / 2) * sy,
                          bw * sx, bh * sy], 1)

        idx = cv2.dnn.NMSBoxes(boxes.tolist(), cf.tolist(), self.conf, self.iou)
        if len(idx) == 0:
            return []
        return [(*boxes[i].astype(int), float(cf[i]))
                for i in np.array(idx).flatten()]


class ThermalStream(threading.Thread):
    """Capture + CLAHE + inference + colormap + encode, in one thread.

    Kept together because the thermal sensor tops out at 25/50fps and the
    frames are tiny, so there is nothing to gain from splitting it the way
    the RGB path is split.
    """

    def __init__(self, target, model, imgsz, conf, cmap, quality,
                 threads, out_w, out_h):
        super().__init__(daemon=True)
        self.target, self.model = target, model
        self.imgsz, self.conf, self.cmap = imgsz, conf, cmap
        self.quality, self.threads = quality, threads
        self.out_w, self.out_h = out_w, out_h
        self.jpeg = Slot()
        self.fps = 0.0
        self.hz = 0.0
        self.ms = 0.0
        self.ndet = 0
        self.running = True

    def run(self):
        cap = open_camera(self.target)
        if cap is None:
            print(f"[Thermal] cannot open camera '{self.target}'")
            return

        pred = None
        if self.model:
            try:
                pred = Predictor(self.model, self.imgsz, self.conf,
                                 threads=self.threads)
                print(f"[Thermal] model ready ({pred.threads} threads) "
                      f"@ {self.imgsz}: {self.model}")
            except Exception as e:
                print(f"[Thermal] model load failed, video only: {e}")

        print(f"[Thermal] streaming from '{self.target}'")
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        cmap = COLORMAPS.get(self.cmap)
        t0 = ti = time.time()
        boxes = []

        while self.running:
            if not cap.grab():
                time.sleep(0.002)
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - t0, 1e-5)
            t0 = now

            # CLAHE greyscale at native resolution — matches the white-hot
            # training data. The colormap is display only; feeding false
            # colour to the model would be a domain mismatch.
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            g = CLAHE.apply(g)

            if pred is not None:
                t = time.time()
                boxes = pred.predict(g)
                self.ms = (time.time() - t) * 1000
                self.ndet = len(boxes)
                n = time.time()
                self.hz = 0.8 * self.hz + 0.2 / max(n - ti, 1e-5)
                ti = n

            up = cv2.resize(g, (self.out_w, self.out_h),
                            interpolation=cv2.INTER_LINEAR)
            disp = cv2.applyColorMap(up, cmap) if cmap is not None \
                else cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)

            sx = self.out_w / g.shape[1]
            sy = self.out_h / g.shape[0]
            for x, y, w, h, c in boxes:
                cv2.rectangle(disp, (int(x * sx), int(y * sy)),
                              (int((x + w) * sx), int((y + h) * sy)),
                              (0, 215, 255), 2)
                cv2.putText(disp, f"{c:.2f}",
                            (int(x * sx), max(int(y * sy) - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

            hud = f"THM {self.fps:.0f}fps | det {self.hz:.0f}Hz {self.ms:.0f}ms | {self.ndet}"
            cv2.rectangle(disp, (5, 5), (400, 32), (15, 15, 15), -1)
            cv2.putText(disp, hud, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            ok, jpg = cv2.imencode('.jpg', disp, enc)
            if ok:
                self.jpeg.put(jpg.tobytes())
        cap.release()


streams = {}


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


HTML = """<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>RDK X5 Teleop</title><style>
body{font-family:system-ui,sans-serif;background:#0b0f19;color:#f8fafc;
margin:0;padding:16px;text-align:center}
h1{color:#38bdf8;font-size:20px;margin:0 0 4px}
p{color:#94a3b8;margin:0 0 14px;font-size:13px}
.g{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
.p{background:#1e293b;padding:10px;border-radius:12px}
img{display:block;border-radius:8px;border:1px solid #334155;max-width:100%}
.l{font-size:11px;letter-spacing:.12em;color:#64748b;margin-top:6px}
</style></head><body>
<h1>RDK X5 Teleop</h1>
<p>RGB drive feed &middot; thermal detection</p>
<div class="g">
  <div class="p"><img src="/rgb_feed" width="640"><div class="l">RGB / NoIR</div></div>
  <div class="p"><img src="/thermal_feed" width="512"><div class="l">THERMAL + YOLO11</div></div>
</div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                      # per-frame logging is not free

    def do_GET(self):
        if self.path == '/':
            body = HTML.encode()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        key = {'/rgb_feed': 'rgb', '/thermal_feed': 'thermal'}.get(self.path)
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
                time.sleep(0.002)      # tight poll keeps latency low
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
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-cam", default="10")
    p.add_argument("--thermal-cam", default="0")
    p.add_argument("--no-thermal", action="store_true",
                   help="Teleop video only, no thermal stream")
    p.add_argument("--no-detect", action="store_true",
                   help="Thermal video without inference")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Thermal model input: 320 ~11Hz, 224 ~24Hz, 192 ~32Hz")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--colormap", choices=list(COLORMAPS), default="inferno")
    p.add_argument("--rgb-width", type=int, default=640)
    p.add_argument("--rgb-height", type=int, default=480)
    p.add_argument("--rgb-fps", type=int, default=60)
    p.add_argument("--rgb-quality", type=int, default=70)
    p.add_argument("--thermal-quality", type=int, default=80)
    p.add_argument("--thermal-out", type=int, nargs=2, default=[512, 384],
                   metavar=("W", "H"))
    p.add_argument("--threads", type=int, default=None,
                   help=f"ONNX threads (default {CORES}, all cores)")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    base = Path(__file__).parent
    model = None
    if not args.no_detect and HAS_ONNX:
        f = base / f"thermal_yolo11n_v3_best_{args.imgsz}.onnx"
        if f.exists():
            model = str(f)
        else:
            fb = base.parent / "thermal_yolo11n_v3/weights/thermal_yolo11n_v3.onnx"
            model = str(fb) if fb.exists() else None
            if model:
                print(f"[!] no export at {args.imgsz}, falling back to {model}")
            else:
                print(f"[!] no thermal model found at imgsz {args.imgsz}")

    print("=" * 56)
    print(f"  {CORES} cores | {args.threads or CORES} threads for thermal")
    print(f"  RGB     cam {args.rgb_cam} @ {args.rgb_width}x{args.rgb_height} "
          f"target {args.rgb_fps}fps, no inference")
    if not args.no_thermal:
        print(f"  Thermal cam {args.thermal_cam} @ imgsz {args.imgsz}")
    print("=" * 56)

    rgb = RGBStream(args.rgb_cam, args.rgb_quality,
                    args.rgb_width, args.rgb_height, args.rgb_fps)
    rgb.start()
    streams['rgb'] = rgb

    if not args.no_thermal:
        thm = ThermalStream(args.thermal_cam, model, args.imgsz, args.conf,
                            args.colormap, args.thermal_quality, args.threads,
                            args.thermal_out[0], args.thermal_out[1])
        thm.start()
        streams['thermal'] = thm

    print(f"\n [OK] http://<rdk-ip>:{args.port}\n")
    srv = ThreadedHTTPServer(('0.0.0.0', args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        for s in streams.values():
            s.running = False
        srv.server_close()


if __name__ == "__main__":
    main()
