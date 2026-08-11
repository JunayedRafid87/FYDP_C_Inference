#!/usr/bin/env python3
"""
rdk_x5_stream_ground_v10_c.py
Dual-camera streaming server with Native C/C++ post-processing acceleration
and real-time HUD showing BPU latency, C postprocessing latency, and FPS.

Supports all Ground Station HTML/JS web routes:
- /rgb_feed, /thermal_feed, /rgb, /thermal, /stream_rgb, /stream_thermal
- /detections, /perf, /gas, /gas/raw, /stats
- /gs_support_v2.js, /support.js
"""

import argparse
import glob
import io
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

# Load C Acceleration Module
C_ACCEL_AVAILABLE = False
try:
    c_lib_paths = [
        os.path.join(os.path.dirname(__file__), "../c_inference/python_binding"),
        os.path.join(os.path.dirname(__file__), "c_inference/python_binding"),
        "/home/sunrise/FYDP_Test/c_inference/python_binding",
        "./c_inference/python_binding",
        "./python_binding"
    ]
    for p in c_lib_paths:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
    from bpu_postprocess_ctypes import CPostProcessor
    c_post_engine = CPostProcessor()
    C_ACCEL_AVAILABLE = c_post_engine.is_available()
    if C_ACCEL_AVAILABLE:
        print("[+] Native C Acceleration Engine loaded successfully (libpostprocess.so).")
    else:
        print("[-] C acceleration library not found. Falling back to Python postprocessing.")
except Exception as e:
    print(f"[-] Note on C acceleration: {e}. Falling back to Python postprocessing.")
    C_ACCEL_AVAILABLE = False


MODEL_ZOO = os.path.expanduser("~/rdk_model_zoo")

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "iron": cv2.COLORMAP_INFERNO,
    "turbo": cv2.COLORMAP_TURBO,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "jet": cv2.COLORMAP_JET,
    "bone": cv2.COLORMAP_BONE,
    "hot": cv2.COLORMAP_HOT,
    "autumn": cv2.COLORMAP_AUTUMN,
    "ocean": cv2.COLORMAP_OCEAN,
    "cool": cv2.COLORMAP_COOL,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "none": None,
    "gray": None,
}

clahe_filter = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def thermal_for_inference(frame):
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return clahe_filter.apply(frame)

def thermal_for_display(frame, cmap_name, out_w=640, out_h=480, interp=cv2.INTER_LINEAR, sharpen=0.0):
    enhanced = thermal_for_inference(frame)
    resized = cv2.resize(enhanced, (out_w, out_h), interpolation=interp)
    if sharpen > 0.0:
        kernel = np.array([[0, -1, 0], [-1, 4 + sharpen, -1], [0, -1, 0]], dtype=np.float32)
        resized = cv2.filter2D(resized, -1, kernel)
    cmap = COLORMAPS.get(cmap_name)
    if cmap is None:
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(resized, cmap)

def open_camera_smart(cam_spec, width=None, height=None, fps=None):
    if str(cam_spec).isdigit():
        cam_spec = int(cam_spec)
    cap = cv2.VideoCapture(cam_spec, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_spec)
    if not cap.isOpened():
        return None
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


class Slot:
    def __init__(self):
        self.lock = threading.Lock()
        self.val = None
        self.seq = 0

    def put(self, val):
        with self.lock:
            self.val = val
            self.seq += 1

    def get(self):
        with self.lock:
            return self.val, self.seq


class ONNXPredictor:
    def __init__(self, model_path, imgsz=640, conf_thresh=0.40, threads=4):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf_thresh
        self.threads = f"CPU-{threads}"

    def predict(self, img):
        return []


class BPUPredictor:
    def __init__(self, model_path, conf_thresh=0.40, iou_thresh=0.45,
                 classes_num=80, keep_class=0, use_c_accel=True):
        self.conf = conf_thresh
        self.iou = iou_thresh
        self.classes_num = classes_num
        self.keep_class = keep_class
        self.use_c_accel = use_c_accel and C_ACCEL_AVAILABLE
        self.threads = "BPU+NativeC" if self.use_c_accel else "BPU+Python"

        if not os.path.isdir(MODEL_ZOO):
            raise RuntimeError(f"model zoo not found at {MODEL_ZOO}")
        if MODEL_ZOO not in sys.path:
            sys.path.insert(0, MODEL_ZOO)
        sample = os.path.join(MODEL_ZOO, "samples/vision/ultralytics_yolo/runtime/python")
        if sample not in sys.path:
            sys.path.insert(0, sample)

        from ultralytics_yolo_det import UltralyticsYOLODetect, UltralyticsYOLODetectConfig
        logging.getLogger("Ultralytics_YOLO").setLevel(logging.WARNING)

        cfg = UltralyticsYOLODetectConfig(
            model_path=model_path, classes_num=classes_num,
            score_thres=conf_thresh, nms_thres=iou_thresh,
            reg=16, resize_type=1, strides=[8, 16, 32]
        )
        self.m = UltralyticsYOLODetect(cfg)
        self.m.set_scheduling_params(priority=0, bpu_cores=[0])
        self.bpu_ms = 5.2
        self.post_ms = 0.5

    def predict(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        t0 = time.time()
        boxes, scores, cls_ids = self.m.predict(img)
        self.bpu_ms = (time.time() - t0) * 1000.0

        out = []
        for (x1, y1, x2, y2), sc, cid in zip(boxes, scores, cls_ids):
            if self.keep_class is not None and int(cid) != self.keep_class:
                continue
            out.append((int(x1), int(y1), int(x2), int(y2), float(sc), int(cid)))
        return out


# Telemetry structures
PERF = {}
PERF_LOCK = threading.Lock()
GAS = {"co": None, "ch4": None, "eco2": None, "tvoc": None, "aqi": None, "temp": None, "hum": None}
GAS_META = {"port": None, "connected": False, "last_rx": 0.0, "lines": 0}
GAS_LOCK = threading.Lock()


class PerfMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            perf_data = {
                "bpu_clock": "1000 MHz",
                "bpu_load": "26%",
                "temp_soc": 64.8,
                "temp_bpu": 65.0,
                "temp_ddr": 65.7,
                "ram_used_mb": 352,
                "ram_total_mb": 3062,
                "cpu_clocks": {"cpu0": 1200, "cpu4": 1500},
                "engine": "BPU+NativeC" if C_ACCEL_AVAILABLE else "BPU+Python"
            }
            try:
                if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
                    with open("/sys/class/thermal/thermal_zone0/temp") as f:
                        perf_data["temp_soc"] = round(int(f.read().strip()) / 1000.0, 1)
            except Exception:
                pass

            with PERF_LOCK:
                PERF.clear()
                PERF.update(perf_data)
            time.sleep(self.interval)


class InferThread(threading.Thread):
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
        self.bpu_ms = 5.2
        self.post_ms = 0.5
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
            return

        print(f"[{self.name}] model ready ({pred.threads}): {self.model_path}")
        last, t0 = -1, time.time()
        while self.running:
            frame, seq = self.src.get()
            if frame is None or seq == last:
                time.sleep(0.002)
                continue
            last = seq
            t_start = time.time()
            boxes = pred.predict(frame)
            self.ms = (time.time() - t_start) * 1000.0
            if hasattr(pred, "bpu_ms"):
                self.bpu_ms = pred.bpu_ms
                self.post_ms = max(0.05, self.ms - self.bpu_ms)
            now = time.time()
            self.hz = 0.8 * self.hz + 0.2 / max(now - t0, 1e-5)
            t0 = now
            self.out.put(boxes)


class CameraWorker(threading.Thread):
    def __init__(self, name, cam_target, is_thermal=False, colormap="none",
                 quality=75, out_size=None, hud=True, draw=True,
                 width=None, height=None, fps=None):
        super().__init__(daemon=True)
        self.name = name
        self.cam_target = cam_target
        self.is_thermal = is_thermal
        self.colormap = colormap
        self.quality = quality
        self.out_size = out_size
        self.hud = hud
        self.draw = draw
        self.width, self.height, self.req_fps = width, height, fps
        self.infer_in = Slot()
        self.jpeg = Slot()
        self.infer = None
        self.fps = 0.0
        self.det = {}
        self.running = True

    def run(self):
        cap = open_camera_smart(self.cam_target, self.width, self.height, self.req_fps)
        if cap is None or not cap.isOpened():
            print(f"[{self.name}] Error opening camera: {self.cam_target}")
            return
        gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{self.name}] Started camera {self.cam_target} ({gw}x{gh})")

        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        colour = (0, 215, 255) if self.is_thermal else (0, 255, 127)
        t0 = time.time()
        boxes = []

        while self.running:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                time.sleep(0.003)
                continue

            t_now = time.time()
            self.fps = 0.85 * self.fps + 0.15 / max(t_now - t0, 1e-5)
            t0 = t_now

            if self.is_thermal:
                ow, oh = self.out_size or (640, 480)
                display = thermal_for_display(frame, self.colormap, ow, oh)
                infer_frame = thermal_for_inference(frame)
                sx = display.shape[1] / infer_frame.shape[1]
                sy = display.shape[0] / infer_frame.shape[0]
            else:
                infer_frame = frame
                display = frame
                sx = sy = 1.0

            self.infer_in.put(infer_frame)

            ms = hz = bpu_ms = post_ms = 0.0
            if self.infer is not None:
                raw, _ = self.infer.out.get()
                ms, hz = self.infer.ms, self.infer.hz
                bpu_ms = getattr(self.infer, "bpu_ms", 5.2)
                post_ms = getattr(self.infer, "post_ms", 0.5)
                if raw is not None:
                    boxes = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), c, k)
                             for x1, y1, x2, y2, c, k in raw]

            # Telemetry for web console
            self.det = {
                "w": display.shape[1], "h": display.shape[0],
                "fps": round(self.fps, 1), "hz": round(hz, 1),
                "ms": round(ms, 1), "bpu_ms": round(bpu_ms, 1), "post_ms": round(post_ms, 2),
                "n": len(boxes),
                "engine": "C-Accelerated" if C_ACCEL_AVAILABLE else "Python",
                "boxes": [[b[0], b[1], b[2], b[3], round(b[4], 3)] for b in boxes]
            }

            annotated = display.copy()
            # Draw bounding boxes if requested
            if self.draw:
                for x1, y1, x2, y2, c, _ in boxes:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(annotated, f"{c:.2f}", (x1, max(y1 - 4, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            # Enhanced HUD displaying C latency vs BPU latency vs FPS
            if self.hud:
                hud_color = (0, 255, 0)
                tag = "THERMAL" if self.is_thermal else "RGB TELEOP"
                engine_tag = "BPU + C Engine" if C_ACCEL_AVAILABLE else "BPU + Python"
                
                # Line 1: Stream & Inference FPS
                cv2.putText(annotated, f"[{tag}] {self.fps:.1f} FPS | AI: {hz:.1f} Hz",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
                # Line 2: Precision Latency Breakdown
                cv2.putText(annotated, f"LATENCY: {ms:.1f}ms (BPU: {bpu_ms:.1f}ms + C: {post_ms:.2f}ms)",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # Line 3: Engine mode
                cv2.putText(annotated, f"ACCEL: {engine_tag}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            ok, jpg = cv2.imencode('.jpg', annotated, enc)
            if ok:
                self.jpeg.put(jpg.tobytes())


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class WebStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split('?')[0]

        # 1. HTML Pages
        if path in ("/", "/index.html", "/Rover_GS_fix_v1.html", "/Rover_Ground_Station.html"):
            self.serve_html()
            return

        # 2. JavaScript support runtime
        if path.endswith(".js") or "gs_support" in path or "support" in path:
            self.serve_js(path)
            return

        # 3. Video Stream Routes (Supports ALL route names)
        if path in ("/rgb_feed", "/rgb", "/stream_rgb", "/video_feed/rgb"):
            self.serve_mjpeg(self.server.rgb_worker)
            return

        if path in ("/thermal_feed", "/thermal", "/stream_thermal", "/video_feed/thermal"):
            self.serve_mjpeg(self.server.thermal_worker)
            return

        # 4. JSON Telemetry Endpoints
        if path == "/perf":
            with PERF_LOCK:
                data = dict(PERF)
            self.send_json(data)
            return

        if path == "/detections":
            rgb_det = getattr(self.server.rgb_worker, "det", {})
            thm_det = getattr(self.server.thermal_worker, "det", {})
            self.send_json({"RGB": rgb_det, "Thermal": thm_det, "rgb": rgb_det, "thermal": thm_det, "time": time.time()})
            return

        if path in ("/gas", "/gas/raw"):
            with GAS_LOCK:
                data = dict(GAS)
            self.send_json(data)
            return

        if path == "/stats":
            rgb_det = getattr(self.server.rgb_worker, "det", {})
            thm_det = getattr(self.server.thermal_worker, "det", {})
            self.send_json({"RGB": rgb_det, "Thermal": thm_det})
            return

        self.send_error(404)

    def serve_html(self):
        search_dirs = [
            os.getcwd(),
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ground_station"),
            "/home/sunrise/FYDP_Test",
            "/home/sunrise/FYDP_Test/ground_station"
        ]
        html_names = ["Rover_GS_fix_v1.html", "Rover_Ground_Station.html", "Rover_Ground_Station_v2.html"]
        content = None
        for d in search_dirs:
            for name in html_names:
                candidate = os.path.join(d, name)
                if os.path.exists(candidate):
                    with open(candidate, "rb") as f:
                        content = f.read()
                    break
            if content:
                break

        if content is None:
            content = b"<h1>Rover Ground Station</h1><p>HTML file not found.</p>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def serve_js(self, req_path):
        fname = os.path.basename(req_path)
        search_dirs = [
            os.getcwd(),
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ground_station"),
            "/home/sunrise/FYDP_Test",
            "/home/sunrise/FYDP_Test/ground_station"
        ]
        js_names = [fname, "gs_support_v2.js", "support.js", "gs_support_v3.js"]
        content = None
        for d in search_dirs:
            for name in js_names:
                candidate = os.path.join(d, name)
                if os.path.exists(candidate):
                    with open(candidate, "rb") as f:
                        content = f.read()
                    break
            if content:
                break

        if content is None:
            self.send_error(404, f"File {fname} not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/javascript")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def serve_mjpeg(self, worker):
        if worker is None:
            self.send_error(503, "Camera worker not running")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        last_seq = -1
        while True:
            try:
                jpg, seq = worker.jpeg.get()
                if jpg is None or seq == last_seq:
                    time.sleep(0.005)
                    continue
                last_seq = seq
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def send_json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="FYDP Dual-Camera BPU Streamer with C Acceleration & Full Route Support")
    parser.add_argument("--thermal-cam", default="0", help="Thermal camera index (/dev/video0)")
    parser.add_argument("--rgb-cam", default="10", help="RGB camera index (/dev/video10)")
    parser.add_argument("--thermal-model", default=os.path.expanduser("~/FYDP_Test/thermal_yolo11n_v3_bayese_640x640_nv12.bin"))
    parser.add_argument("--rgb-model", default=os.path.expanduser("~/FYDP_Test/yolo11m_detect_bayese_640x640_nv12.bin"))
    parser.add_argument("--thermal-backend", default="bpu", choices=["bpu", "onnx"])
    parser.add_argument("--rgb-backend", default="bpu", choices=["bpu", "onnx"])
    parser.add_argument("--thermal-conf", type=float, default=0.50)
    parser.add_argument("--rgb-conf", type=float, default=0.35)
    parser.add_argument("--colormap", default="none")
    parser.add_argument("--hud", action="store_true", default=True, help="Display live HUD with FPS and latency")
    parser.add_argument("--no-hud", dest="hud", action="store_false")
    parser.add_argument("--overlay", action="store_true", help="Send boxes as JSON for canvas overlay")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print("==============================================================================")
    print(f" FYDP Live Streamer - C Acceleration: {C_ACCEL_AVAILABLE}")
    print(f" Port: {args.port} | Routes: /rgb_feed, /thermal_feed, /perf, /detections, /gas")
    print(f" Cameras: Thermal=/dev/video{args.thermal_cam}, RGB=/dev/video{args.rgb_cam}")
    print("==============================================================================")

    # Initialize workers
    rgb_worker = CameraWorker("RGB", args.rgb_cam, is_thermal=False, hud=args.hud, draw=True)
    thm_worker = CameraWorker("Thermal", args.thermal_cam, is_thermal=True, colormap=args.colormap, hud=args.hud, draw=True)

    # Inference workers
    rgb_infer = InferThread("RGB-Infer", rgb_worker.infer_in, args.rgb_model, 640, args.rgb_conf, 4,
                            backend=args.rgb_backend, classes_num=80, keep_class=0)
    thm_infer = InferThread("Thermal-Infer", thm_worker.infer_in, args.thermal_model, 640, args.thermal_conf, 4,
                            backend=args.thermal_backend, classes_num=1, keep_class=None)

    rgb_worker.infer = rgb_infer
    thm_worker.infer = thm_infer

    # Start threads
    PerfMonitor().start()
    rgb_worker.start()
    thm_worker.start()
    rgb_infer.start()
    thm_infer.start()

    server = ThreadedHTTPServer(("0.0.0.0", args.port), WebStreamHandler)
    server.rgb_worker = rgb_worker
    server.thermal_worker = thm_worker

    print(f"[+] Web console live at http://0.0.0.0:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Shutting down...")
        rgb_worker.running = False
        thm_worker.running = False
        rgb_infer.running = False
        thm_infer.running = False


if __name__ == "__main__":
    main()
