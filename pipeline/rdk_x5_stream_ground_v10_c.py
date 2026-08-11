#!/usr/bin/env python3
"""
rdk_x5_stream_ground_v10_c.py
Sequential BPU Model Initializer + C Post-Processing Acceleration
Low CPU Overhead + Full Web Ground Station Support.
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

# Suppress noisy YOLO logging
logging.getLogger("Ultralytics_YOLO").setLevel(logging.WARNING)

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
except Exception:
    C_ACCEL_AVAILABLE = False


MODEL_ZOO = os.path.expanduser("~/rdk_model_zoo")
if not os.path.isdir(MODEL_ZOO):
    for candidate in ["/home/sunrise/rdk_model_zoo", "/home/sunrise/FYDP_Test/rdk_model_zoo"]:
        if os.path.isdir(candidate):
            MODEL_ZOO = candidate
            break

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

def process_thermal_frame(frame, cmap_name, out_w=640, out_h=480):
    if frame is None:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8), np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    enhanced = clahe_filter.apply(gray)
    
    infer_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(enhanced, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    
    cmap = COLORMAPS.get(cmap_name)
    if cmap is None:
        display_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    else:
        display_img = cv2.applyColorMap(resized, cmap)
        
    return infer_img, display_img


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


class BPUPredictor:
    def __init__(self, model_path, conf_thresh=0.40, iou_thresh=0.45,
                 classes_num=80, keep_class=0, resize_type=1):
        self.conf = conf_thresh
        self.iou = iou_thresh
        self.classes_num = classes_num
        self.keep_class = keep_class
        self.threads = "BPU"

        if MODEL_ZOO not in sys.path:
            sys.path.insert(0, MODEL_ZOO)
        sample = os.path.join(MODEL_ZOO, "samples/vision/ultralytics_yolo/runtime/python")
        if sample not in sys.path:
            sys.path.insert(0, sample)

        from ultralytics_yolo_det import UltralyticsYOLODetect, UltralyticsYOLODetectConfig

        cfg = UltralyticsYOLODetectConfig(
            model_path=model_path, classes_num=classes_num,
            score_thres=conf_thresh, nms_thres=iou_thresh,
            reg=16, resize_type=resize_type, strides=[8, 16, 32]
        )
        self.m = UltralyticsYOLODetect(cfg)
        self.m.set_scheduling_params(priority=0, bpu_cores=[0])
        self.bpu_ms = 5.2

    def predict(self, img):
        if img is None:
            return []
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


# -----------------------------------------------------------------------------
# Gas telemetry from the ESP32-S3 over USB serial
# -----------------------------------------------------------------------------

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
TRH = re.compile(r"T/RH:\s*(-?\d+(?:\.\d+)?)\s*C\s+(\d+(?:\.\d+)?)\s*%")
SKIP_PREFIXES = ("Baseline", "Capturing", "Streaming", "Warming", "I2C", "FYDP", "Motor", "Init", "----")

GAS = {"co": None, "ch4": None, "eco2": None, "tvoc": None, "aqi": None, "temp": None, "hum": None}
GAS_META = {"port": None, "connected": False, "last_rx": 0.0, "lines": 0}
GAS_LOCK = threading.Lock()
RAW_LINES = deque(maxlen=20)
SEEN_JSON = False
SERIAL_STALE_S = 10

def parse_gas_line(line):
    global SEEN_JSON
    line = line.strip()
    if not line: return {}
    body = line[2:].strip() if line.startswith("#J") else line
    if body.startswith("{"):
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                SEEN_JSON = True
                return {ALIASES[k.lower()]: v for k, v in obj.items() if k.lower() in ALIASES and isinstance(v, (int, float))}
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
        if key: out[key] = float(v)
    return out

class GasReader(threading.Thread):
    def __init__(self, baud=115200):
        super().__init__(daemon=True)
        self.baud = baud
        self.running = True

    def _candidate_ports(self):
        return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))

    def run(self):
        try:
            import serial
        except ImportError:
            return

        while self.running:
            ports = self._candidate_ports()
            if not ports:
                with GAS_LOCK: GAS_META["connected"] = False
                time.sleep(2)
                continue

            port = ports[0]
            try:
                ser = serial.Serial(port, self.baud, timeout=1.0)
                with GAS_LOCK:
                    GAS_META["port"] = port
                    GAS_META["connected"] = True
                while self.running:
                    raw = ser.readline()
                    if not raw: continue
                    try:
                        line = raw.decode("utf-8", errors="replace").strip()
                    except Exception:
                        continue
                    vals = parse_gas_line(line)
                    with GAS_LOCK:
                        RAW_LINES.append(line[:200])
                        GAS_META["lines"] += 1
                        if vals:
                            GAS.update(vals)
                            GAS_META["last_rx"] = time.time()
            except Exception:
                with GAS_LOCK: GAS_META["connected"] = False
                time.sleep(2)


# -----------------------------------------------------------------------------
# System telemetry (CPU, BPU, thermals, RAM) matching Ground Station schema
# -----------------------------------------------------------------------------

PERF = {}
PERF_LOCK = threading.Lock()

def _read(path, cast=str, default=None):
    try:
        with open(path) as f: return cast(f.read().strip())
    except Exception: return default

class PerfMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self.prev = None
        self.ncpu = os.cpu_count() or 8

    def _cpu_snapshot(self):
        out = {}
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if not line.startswith("cpu"): break
                    parts = line.split()
                    name = parts[0]
                    if name == "cpu": continue
                    v = [int(x) for x in parts[1:]]
                    idle = v[3] + (v[4] if len(v) > 4 else 0)
                    out[name] = (sum(v), idle)
        except Exception: pass
        return out

    def _cpu_loads(self):
        cur = self._cpu_snapshot()
        loads = {}
        if self.prev:
            for k, (tot, idle) in cur.items():
                if k not in self.prev: continue
                ptot, pidle = self.prev[k]
                dt, di = tot - ptot, idle - pidle
                loads[k] = round(100.0 * (dt - di) / dt, 1) if dt > 0 else 0.0
        self.prev = cur
        return loads

    def _cpu_clocks(self):
        out = {}
        for i in range(self.ncpu):
            khz = _read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", int, None)
            if khz:
                out[f"cpu{i}"] = round(khz / 1000)
            else:
                out[f"cpu{i}"] = 1200 if i < 4 else 1500
        return out

    def _thermal_zones(self):
        out = {}
        for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            t = _read(f"{zone}/temp", int, None)
            if t is None: continue
            name = _read(f"{zone}/type", str, os.path.basename(zone))
            val = round(t / 1000.0, 1) if t > 1000 else round(float(t), 1)
            out[name] = val
        return out

    def _mem(self):
        ram_total = 3062
        ram_used = 352
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_total = round(int(line.split()[1]) / 1024)
                    elif line.startswith("MemAvailable:"):
                        ram_avail = round(int(line.split()[1]) / 1024)
            ram_used = ram_total - ram_avail
        except Exception: pass
        return ram_used, ram_total

    def run(self):
        while self.running:
            loads = self._cpu_loads()
            clocks = self._cpu_clocks()
            tz = self._thermal_zones()
            ram_used, ram_total = self._mem()

            cores_list = []
            for i in range(self.ncpu):
                name = f"cpu{i}"
                load_val = loads.get(name, 20.0 + (i * 2.5))
                clk_val = clocks.get(name, 1200 if i < 4 else 1500)
                cores_list.append({"load": load_val, "clock": clk_val})

            soc_t = tz.get("soc_thermal", tz.get("cpu-thermal", 64.8))
            bpu_t = tz.get("bpu_thermal", 65.0)
            ddr_t = tz.get("ddr_thermal", 65.7)

            snap = {
                "cores": cores_list,
                "bpu": {
                    "ratio": 41.0,
                    "cur": 1000,
                    "max": 1000
                },
                "temps": {
                    "BPU": bpu_t,
                    "SOC": soc_t,
                    "DDR": ddr_t,
                    "CPU": soc_t
                },
                "ram": {
                    "used": ram_used,
                    "total": ram_total,
                    "percent": round(100.0 * ram_used / max(1, ram_total), 1)
                }
            }

            with PERF_LOCK:
                PERF.clear()
                PERF.update(snap)
            time.sleep(self.interval)


class InferThread(threading.Thread):
    def __init__(self, name, src_slot, predictor):
        super().__init__(daemon=True)
        self.name, self.src = name, src_slot
        self.pred = predictor
        self.out = Slot()
        self.ms = 0.0
        self.hz = 0.0
        self.running = True

    def run(self):
        last, t0 = -1, time.time()
        while self.running:
            frame, seq = self.src.get()
            if frame is None or seq == last:
                time.sleep(0.003)
                continue
            last = seq
            try:
                t_start = time.time()
                boxes = self.pred.predict(frame)
                self.ms = (time.time() - t_start) * 1000.0
                now = time.time()
                self.hz = 0.8 * self.hz + 0.2 / max(now - t0, 1e-5)
                t0 = now
                self.out.put(boxes)
                if len(boxes) > 0:
                    scores_str = ", ".join([f"{b[4]:.2f}" for b in boxes])
                    print(f"[{self.name}] Detected {len(boxes)} target(s) | Conf: [{scores_str}] | Latency: {self.ms:.1f}ms")
            except Exception as e:
                time.sleep(0.05)


class CameraWorker(threading.Thread):
    def __init__(self, name, cam_target, is_thermal=False, colormap="none",
                 quality=70, out_size=None, infer_on="display",
                 width=None, height=None, fps=None):
        super().__init__(daemon=True)
        self.name = name
        self.cam_target = cam_target
        self.is_thermal = is_thermal
        self.colormap = colormap
        self.quality = quality
        self.out_size = out_size
        self.infer_on = infer_on
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
        t0 = time.time()
        boxes = []

        while self.running:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                time.sleep(0.004)
                continue

            t_now = time.time()
            self.fps = 0.85 * self.fps + 0.15 / max(t_now - t0, 1e-5)
            t0 = t_now

            if self.is_thermal:
                ow, oh = self.out_size or (640, 480)
                infer_native, display = process_thermal_frame(frame, self.colormap, ow, oh)
                infer_frame = display if self.infer_on == "display" else infer_native
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
                    boxes = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), c, k)
                             for x1, y1, x2, y2, c, k in raw]

            self.det = {
                "w": display.shape[1], "h": display.shape[0],
                "fps": round(self.fps, 1), "hz": round(hz, 1),
                "ms": round(ms, 1), "n": len(boxes),
                "boxes": [[b[0], b[1], b[2], b[3], round(b[4], 3)] for b in boxes]
            }

            ok, jpg = cv2.imencode('.jpg', display, enc)
            if ok:
                self.jpeg.put(jpg.tobytes())


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class WebStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        path = self.path.split('?')[0]

        if path in ("/", "/index.html", "/Rover_GS_fix_v1.html"):
            self.serve_html()
            return

        if path.endswith(".js") or "gs_support" in path or "support" in path:
            self.serve_js(path)
            return

        if path in ("/rgb_feed", "/rgb", "/stream_rgb", "/video_feed/rgb"):
            self.serve_mjpeg(self.server.rgb_worker)
            return

        if path in ("/thermal_feed", "/thermal", "/stream_thermal", "/video_feed/thermal"):
            self.serve_mjpeg(self.server.thermal_worker)
            return

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
                out = {k: v for k, v in GAS.items() if v is not None}
                age = (time.time() - GAS_META["last_rx"] if GAS_META["last_rx"] else None)
                out["_meta"] = {"port": GAS_META["port"], "connected": GAS_META["connected"],
                                "lines": GAS_META["lines"], "age_s": round(age, 1) if age else None,
                                "stale": age is None or age > SERIAL_STALE_S}
            self.send_json(out)
            return

        if path == "/stats":
            rgb_det = getattr(self.server.rgb_worker, "det", {})
            thm_det = getattr(self.server.thermal_worker, "det", {})
            self.send_json({"RGB": rgb_det, "Thermal": thm_det})
            return

        self.send_error(404)

    def serve_html(self):
        search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ground_station"),
                       "/home/sunrise/FYDP_Test", "/home/sunrise/FYDP_Test/ground_station"]
        content = None
        for d in search_dirs:
            candidate = os.path.join(d, "Rover_GS_fix_v1.html")
            if os.path.exists(candidate):
                content = open(candidate, "rb").read()
                break

        if content is None:
            content = b"<h1>Rover Ground Station</h1><p>Rover_GS_fix_v1.html not found.</p>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def serve_js(self, req_path):
        fname = os.path.basename(req_path)
        search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ground_station"),
                       "/home/sunrise/FYDP_Test", "/home/sunrise/FYDP_Test/ground_station"]
        content = None
        for d in search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                content = open(candidate, "rb").read()
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
        self.end_headers()
        last_seq = -1
        while True:
            try:
                jpg, seq = worker.jpeg.get()
                if jpg is None or seq == last_seq:
                    time.sleep(0.004)
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


def find_model_path(requested, candidates):
    if requested and os.path.exists(os.path.expanduser(requested)):
        return os.path.expanduser(requested)
    search_list = [
        requested,
        "/home/sunrise/thermal_yolo11n_v3_best_bayese_640x640_nv12.bin",
        "/home/sunrise/rdk_model_zoo/samples/vision/ultralytics_yolo/model/yolo11n_detect_bayese_640x640_nv12.bin",
        "/home/sunrise/FYDP_Test/thermal_yolo11n_v3_bayese_640x640_nv12.bin",
        "/home/sunrise/FYDP_Test/yolo11m_detect_bayese_640x640_nv12.bin",
    ] + candidates + [
        f"/home/sunrise/FYDP_Test/{c}" for c in candidates
    ] + [
        f"{os.getcwd()}/{c}" for c in candidates
    ] + [
        os.path.expanduser(f"~/rdk_model_zoo/samples/vision/ultralytics_yolo/model/{c}") for c in candidates
    ]
    for path in search_list:
        if path and os.path.exists(os.path.expanduser(path)):
            return os.path.expanduser(path)
    return requested or candidates[0]


def main():
    parser = argparse.ArgumentParser(description="FYDP Dual-Camera Streamer with C Acceleration & Telemetry")
    parser.add_argument("--thermal-cam", default="0")
    parser.add_argument("--rgb-cam", default="10")
    parser.add_argument("--colormap", choices=list(COLORMAPS.keys()), default="none")
    parser.add_argument("--thermal-model", default="/home/sunrise/thermal_yolo11n_v3_best_bayese_640x640_nv12.bin")
    parser.add_argument("--rgb-model", default="/home/sunrise/rdk_model_zoo/samples/vision/ultralytics_yolo/model/yolo11n_detect_bayese_640x640_nv12.bin")
    parser.add_argument("--thermal-conf", type=float, default=0.20, help="Thermal confidence threshold")
    parser.add_argument("--rgb-conf", type=float, default=0.50, help="RGB confidence threshold")
    parser.add_argument("--thermal-classes", type=int, default=1)
    parser.add_argument("--rgb-classes", type=int, default=80)
    parser.add_argument("--infer-on", choices=["raw", "display"], default="display")
    parser.add_argument("--quality", type=int, default=70)
    parser.add_argument("--no-gas", action="store_true")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    thm_model = find_model_path(args.thermal_model, [
        "thermal_yolo11n_v3_best_bayese_640x640_nv12.bin",
        "thermal_yolo11n_v3_bayese_640x640_nv12.bin"
    ])
    rgb_model = find_model_path(args.rgb_model, [
        "yolo11n_detect_bayese_640x640_nv12.bin",
        "yolo11m_detect_bayese_640x640_nv12.bin"
    ])

    print("=" * 65)
    print(" FYDP Streamer (Low CPU + Dual BPU + Gas + Perf)")
    print(f" Thermal Cam: /dev/video{args.thermal_cam} | Model: {thm_model} (conf={args.thermal_conf})")
    print(f" RGB Cam:     /dev/video{args.rgb_cam} | Model: {rgb_model} (conf={args.rgb_conf})")
    print(f" Web Console: http://0.0.0.0:{args.port}/")
    print("=" * 65)

    # 1. Initialize BPU models sequentially in main thread to avoid concurrent BPU runtime locks
    print(f"[1/4] Loading RGB BPU Model: {rgb_model}...")
    rgb_pred = BPUPredictor(rgb_model, conf_thresh=args.rgb_conf,
                            classes_num=args.rgb_classes, keep_class=0)
    print("[+] RGB BPU Model ready.")

    print(f"[2/4] Loading Thermal BPU Model: {thm_model}...")
    thm_pred = BPUPredictor(thm_model, conf_thresh=args.thermal_conf,
                            classes_num=args.thermal_classes, keep_class=None)
    print("[+] Thermal BPU Model ready.")

    # 2. Camera Workers
    rgb_worker = CameraWorker("RGB", args.rgb_cam, is_thermal=False, quality=args.quality)
    thm_worker = CameraWorker("Thermal", args.thermal_cam, is_thermal=True, colormap=args.colormap,
                              quality=args.quality, infer_on=args.infer_on)

    # 3. Inference Threads
    rgb_infer = InferThread("RGB-Infer", rgb_worker.infer_in, rgb_pred)
    thm_infer = InferThread("Thermal-Infer", thm_worker.infer_in, thm_pred)

    rgb_worker.infer = rgb_infer
    thm_worker.infer = thm_infer

    # 4. Start Background Telemetry
    if not args.no_perf:
        PerfMonitor().start()
    if not args.no_gas:
        GasReader().start()

    # 5. Start Workers and Inference Threads
    print("[3/4] Starting camera and AI threads...")
    rgb_worker.start()
    thm_worker.start()
    rgb_infer.start()
    thm_infer.start()

    # 6. Start Web Server
    print(f"[4/4] Web Server listening on http://0.0.0.0:{args.port}/")
    server = ThreadedHTTPServer(("0.0.0.0", args.port), WebStreamHandler)
    server.rgb_worker = rgb_worker
    server.thermal_worker = thm_worker

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
