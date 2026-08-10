#!/usr/bin/env python3
"""
RDK X5 - Dual camera teleop + async detection.

ARCHITECTURE
    Per camera, two threads:

      CaptureThread   grab -> (CLAHE for thermal) -> publish frame for the
                      model -> draw NEWEST available boxes -> JPEG -> publish
      InferThread     take the newest frame, run the model, publish boxes

    Video is never blocked by inference. Boxes are drawn from whatever the
    model last produced, onto the current frame. Two independent HTTP
    endpoints so a slow stream cannot hold back a fast one.

WHY ASYNC BEATS "DETECT EVERY Nth FRAME"
    A fixed stride detects on frames 0, 3, 6... whether or not the model is
    ready, so it either wastes capacity or falls behind. Async detection
    always picks the newest frame the instant the model frees up: same CPU
    cost, fresher boxes, no tuning.

WHAT TO EXPECT ON THIS BOARD
    RGB video        30-60fps at 640x480
    Thermal video    25fps (sensor offers 25 or 50 only)
    Latency          ~100ms glass-to-browser
    Detection        ~5-6Hz per model with both running
                     ~11Hz thermal alone at imgsz 320
                     ~32Hz thermal alone at imgsz 192

    Measured pure inference at 320: 1thr 436ms / 2thr 235ms / 4thr 133ms /
    8thr 86ms. Two models on 8 cores is a ~170ms floor, so 30Hz detection
    on both is not reachable on CPU. Sub-50ms latency is not reachable with
    MJPEG in a browser either — decode and display buffering exceed that.

Usage:
    python3 rdk_x5_teleop.py --thermal-cam 0 --rgb-cam 10
    python3 rdk_x5_teleop.py --thermal-cam 0 --rgb-cam 10 --no-rgb-detect
    python3 rdk_x5_teleop.py --thermal-cam 0 --rgb-cam 10 --imgsz 192

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

# Degenerate-box guards. All tunable from the CLI — defaults are
# deliberately permissive on MAX_BOX_DIM because a person at close range
# legitimately fills the frame. The training pipeline's 0.92 was a
# DATASET filter; applying it at inference discards real detections.
MAX_BOX_DIM = 0.98      # only reject literal full-frame boxes
MIN_DIM_FRAC = 0.04     # reject slivers that render as bare lines
MIN_ASPECT = 0.15       # w/h floor: a person is not a vertical hairline
MAX_ASPECT = 4.0        # w/h ceiling
MAX_DET = 10
DEBUG = False


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


class Predictor:
    def __init__(self, path, imgsz, conf=0.40, iou=0.45, threads=None):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or max(2, CORES // 2)
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(path, opts,
                                      providers=['CPUExecutionProvider'])
        self.inp = self.s.get_inputs()[0].name
        self.outs = [o.name for o in self.s.get_outputs()]
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.threads = opts.intra_op_num_threads
        self.tag = Path(path).stem[:14]

    def predict(self, img):
        """Postprocess deliberately mirrors the version that was verified
        working on this hardware: float32 normalise BEFORE transpose, and
        INTEGER boxes into NMS. Also applies the same degenerate-box guards
        the training pipeline used (MAX_BOX_DIM=0.92), which catch the
        zero-width slivers and full-frame boxes that otherwise render as
        stray lines."""
        h, w = img.shape[:2]
        resized = cv2.resize(img, (self.imgsz, self.imgsz))
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]

        preds = self.s.run(self.outs, {self.inp: blob})[0][0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        sc = preds[:, 4:]
        cf = sc[np.arange(len(sc)), sc.argmax(1)]
        keep = cf >= self.conf
        if not keep.any():
            return []
        p, cf = preds[keep], cf[keep]

        sx, sy = w / self.imgsz, h / self.imgsz
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        bw_s, bh_s = bw * sx, bh * sy

        # Degenerate-box guards. MAX_BOX_DIM mirrors the training filter;
        # MIN_DIM_FRAC kills the zero-width slivers that draw as lines.
        aspect = bw_s / np.maximum(bh_s, 1e-6)
        ok = ((bw_s > MIN_DIM_FRAC * w) & (bh_s > MIN_DIM_FRAC * h) &
              (bw_s < MAX_BOX_DIM * w) & (bh_s < MAX_BOX_DIM * h) &
              (aspect > MIN_ASPECT) & (aspect < MAX_ASPECT))
        if DEBUG:
            print(f"  [{self.tag}] conf-pass {len(p)} -> guards {int(ok.sum())}")
            for i in range(min(len(p), 12)):
                print(f"     w={bw_s[i]:6.1f} h={bh_s[i]:6.1f} "
                      f"ar={aspect[i]:5.2f} conf={cf[i]:.3f} "
                      f"{'KEEP' if ok[i] else 'drop'}")
        if not ok.any():
            return []
        p, cf = p[ok], cf[ok]
        cx, cy = p[:, 0], p[:, 1]
        bw_s, bh_s = p[:, 2] * sx, p[:, 3] * sy

        boxes = [[int((cx[i] - p[i, 2] / 2) * sx),
                  int((cy[i] - p[i, 3] / 2) * sy),
                  int(bw_s[i]), int(bh_s[i])] for i in range(len(p))]

        idx = cv2.dnn.NMSBoxes(boxes, cf.tolist(), self.conf, self.iou)
        if len(idx) == 0:
            return []
        order = np.array(idx).flatten()
        order = order[np.argsort(-cf[order])][:MAX_DET]
        return [(*boxes[i], float(cf[i])) for i in order]


class InferThread(threading.Thread):
    """Consumes the newest frame only. Never queues, never falls behind."""

    def __init__(self, name, src_slot, model, imgsz, conf, threads):
        super().__init__(daemon=True)
        self.name, self.src = name, src_slot
        self.model, self.imgsz = model, imgsz
        self.conf, self.threads = conf, threads
        self.out = Slot()
        self.ms = 0.0
        self.hz = 0.0
        self.running = True

    def run(self):
        try:
            pred = Predictor(self.model, self.imgsz, self.conf, self.threads)
        except Exception as e:
            print(f"[{self.name}] model load failed: {e}")
            return
        print(f"[{self.name}] model ready ({pred.threads} threads) "
              f"@ {self.imgsz}: {self.model}")

        last, t0 = -1, time.time()
        while self.running:
            frame, seq = self.src.get()
            if frame is None or seq == last:
                time.sleep(0.003)
                continue
            last = seq

            t = time.time()
            b = pred.predict(frame)
            self.ms = (time.time() - t) * 1000
            now = time.time()
            self.hz = 0.8 * self.hz + 0.2 / max(now - t0, 1e-5)
            t0 = now

            # Boxes travel with the shape they were computed against, so the
            # drawing side can rescale correctly even though the display
            # frame is a different size.
            self.out.put((b, frame.shape[:2]))


class CameraThread(threading.Thread):
    """Capture, publish for inference, draw newest boxes, encode."""

    def __init__(self, name, target, is_thermal, cmap, quality,
                 out_size=None, width=None, height=None, fps=None,
                 colour=(0, 255, 127), infer_color=True):
        super().__init__(daemon=True)
        self.name, self.target = name, target
        self.is_thermal, self.cmap = is_thermal, cmap
        self.quality, self.out_size = quality, out_size
        self.width, self.height, self.req_fps = width, height, fps
        self.colour = colour
        self.infer_color = infer_color
        self.infer_in = Slot()
        self.jpeg = Slot()
        self.infer = None          # set by main if detection is enabled
        self.fps = 0.0
        self.enc_ms = 0.0
        self.running = True

    def run(self):
        cap = open_camera(self.target, self.width, self.height, self.req_fps)
        if cap is None:
            print(f"[{self.name}] cannot open camera '{self.target}'")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{self.name}] streaming {w}x{h} from '{self.target}'")

        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        cmap = COLORMAPS.get(self.cmap)
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

            if self.is_thermal:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                g = CLAHE.apply(g)
                ow, oh = self.out_size or (g.shape[1] * 2, g.shape[0] * 2)
                up = cv2.resize(g, (ow, oh), interpolation=cv2.INTER_LINEAR)
                disp = cv2.applyColorMap(up, cmap) if cmap is not None \
                    else cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
                # MEASURED, not assumed: the model behaves far better on the
                # upscaled colormapped frame than on plain CLAHE greyscale.
                # Greyscale produced tens of false boxes. Whatever the
                # training pipeline actually did, false colour matches it.
                # --thermal-infer gray to test the other path.
                self.infer_in.put(disp if self.infer_color else g)
            else:
                self.infer_in.put(frame)
                disp = frame

            ndet, hz, ms = 0, 0.0, 0.0
            if self.infer is not None:
                payload, _ = self.infer.out.get()
                hz, ms = self.infer.hz, self.infer.ms
                if payload:
                    boxes, box_shape = payload
                    ndet = len(boxes)
                    sy = disp.shape[0] / box_shape[0]
                    sx = disp.shape[1] / box_shape[1]
                    if disp is frame:
                        disp = frame.copy()
                    for x, y, bw, bh, c in boxes:
                        cv2.rectangle(disp,
                                      (int(x * sx), int(y * sy)),
                                      (int((x + bw) * sx), int((y + bh) * sy)),
                                      self.colour, 2)
                        cv2.putText(disp, f"{c:.2f}",
                                    (int(x * sx), max(int(y * sy) - 6, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    self.colour, 2)

            if disp is frame:
                disp = frame.copy()
            hud = (f"{self.name} {self.fps:.0f}fps"
                   + (f" | det {hz:.0f}Hz {ms:.0f}ms | {ndet}"
                      if self.infer is not None else " | no detect"))
            cv2.rectangle(disp, (5, 5), (5 + 8 * len(hud), 30), (15, 15, 15), -1)
            cv2.putText(disp, hud, (11, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 1, cv2.LINE_AA)

            te = time.time()
            ok, jpg = cv2.imencode('.jpg', disp, enc)
            self.enc_ms = (time.time() - te) * 1000
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
<h1>RDK X5 Dual-Modality Detection</h1>
<p>Independent streams &middot; detection asynchronous</p>
<div class="g">
  <div class="p"><img src="/rgb_feed" width="640"><div class="l">RGB / NoIR</div></div>
  <div class="p"><img src="/thermal_feed" width="512"><div class="l">THERMAL</div></div>
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

        key = {'/rgb_feed': 'RGB', '/thermal_feed': 'THM'}.get(self.path)
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


def find_model(base, prefix, imgsz, fallback):
    f = base / f"{prefix}_{imgsz}.onnx"
    if f.exists():
        return str(f)
    fb = base.parent / fallback
    if fb.exists():
        print(f"[!] no export at {imgsz} for {prefix}, using {fb.name}")
        return str(fb)
    print(f"[!] no model found for {prefix} at imgsz {imgsz}")
    return None


def main():
    global MAX_BOX_DIM, MIN_DIM_FRAC, DEBUG
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-cam", default="10")
    p.add_argument("--thermal-cam", default="0")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Thermal input: 320 ~11Hz alone, 224 ~24Hz, 192 ~32Hz")
    p.add_argument("--rgb-imgsz", type=int, default=None)
    p.add_argument("--no-rgb-detect", action="store_true",
                   help="RGB video only — frees the whole CPU for thermal")
    p.add_argument("--no-thermal-detect", action="store_true")
    p.add_argument("--conf", type=float, default=0.45,
                   help="Model summary gives 0.45 as the F1-optimal point")
    p.add_argument("--colormap", choices=list(COLORMAPS), default="inferno")
    p.add_argument("--thermal-infer", choices=["color", "gray"], default="color",
                   help="What the thermal model sees. 'color' = upscaled "
                        "colormapped frame (measured to work). 'gray' = plain "
                        "CLAHE greyscale (produced many false boxes).")
    p.add_argument("--rgb-width", type=int, default=640)
    p.add_argument("--rgb-height", type=int, default=480)
    p.add_argument("--rgb-fps", type=int, default=60)
    p.add_argument("--rgb-quality", type=int, default=70)
    p.add_argument("--thermal-quality", type=int, default=80)
    p.add_argument("--thermal-out", type=int, nargs=2, default=[512, 384],
                   metavar=("W", "H"))
    p.add_argument("--threads", type=int, default=None,
                   help="ONNX threads per model. Default: half the cores when "
                        "both models run, all cores when only one does.")
    p.add_argument("--max-box", type=float, default=MAX_BOX_DIM,
                   help="Reject boxes larger than this fraction of the frame")
    p.add_argument("--min-box", type=float, default=MIN_DIM_FRAC,
                   help="Reject boxes smaller than this fraction (kills slivers)")
    p.add_argument("--debug", action="store_true",
                   help="Print every surviving box with dims, aspect and conf")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    MAX_BOX_DIM, MIN_DIM_FRAC, DEBUG = args.max_box, args.min_box, args.debug

    base = Path(__file__).parent
    rgb_sz = args.rgb_imgsz or args.imgsz

    want_rgb = not args.no_rgb_detect and HAS_ONNX
    want_thm = not args.no_thermal_detect and HAS_ONNX
    n_models = int(want_rgb) + int(want_thm)
    threads = args.threads or (CORES if n_models <= 1 else max(2, CORES // 2))

    rgb_model = find_model(base, "rgb_yolo11n_best", rgb_sz,
                           "rgb_yolo11n/weights/rgb_yolo11n_best.onnx") \
        if want_rgb else None
    thm_model = find_model(base, "thermal_yolo11n_v3_best", args.imgsz,
                           "thermal_yolo11n_v3/weights/thermal_yolo11n_v3.onnx") \
        if want_thm else None

    print("=" * 58)
    print(f"  {CORES} cores | {n_models} model(s) | {threads} threads each")
    print(f"  RGB     cam {args.rgb_cam} @ {args.rgb_width}x{args.rgb_height} "
          f"target {args.rgb_fps}fps"
          + (f", detect @ {rgb_sz}" if rgb_model else ", no detect"))
    print(f"  Thermal cam {args.thermal_cam}"
          + (f", detect @ {args.imgsz} on {args.thermal_infer}"
             if thm_model else ", no detect"))
    print("=" * 58)

    rgb = CameraThread("RGB", args.rgb_cam, False, None, args.rgb_quality,
                       width=args.rgb_width, height=args.rgb_height,
                       fps=args.rgb_fps, colour=(0, 255, 127))
    thm = CameraThread("THM", args.thermal_cam, True, args.colormap,
                       args.thermal_quality, out_size=tuple(args.thermal_out),
                       colour=(0, 215, 255),
                       infer_color=(args.thermal_infer == "color"))

    if rgb_model:
        rgb.infer = InferThread("RGB", rgb.infer_in, rgb_model, rgb_sz,
                                args.conf, threads)
        rgb.infer.start()
    if thm_model:
        thm.infer = InferThread("THM", thm.infer_in, thm_model, args.imgsz,
                                args.conf, threads)
        thm.infer.start()

    rgb.start()
    thm.start()
    streams["RGB"], streams["THM"] = rgb, thm

    print(f"\n [OK] http://<rdk-ip>:{args.port}\n")
    srv = ThreadedHTTPServer(('0.0.0.0', args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        for s in (rgb, thm):
            s.running = False
            if s.infer:
                s.infer.running = False
        srv.server_close()


if __name__ == "__main__":
    main()
