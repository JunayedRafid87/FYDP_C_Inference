#!/usr/bin/env python3
"""
rover_cams.py — ground station page, both camera feeds, and gas telemetry.

    RGB     : IMX219 NoIR, MIPI CSI, NV12, JPEG
    Thermal : UVC module, USB, YUYV 256x192, WHITE-HOT greyscale, LOSSLESS PNG
    Gas     : ESP32-S3 over USB serial (/dev/ttyACM*), JSON or key=value lines

Endpoints
    /           the ground station page
    /rgb        MJPEG stream
    /thermal    MPNG stream (lossless)
    /config     GET/POST fps and sharpness per feed
    /gas        latest parsed sensor values
    /gas/raw    last 20 raw serial lines — use this to see what the ESP32
                is actually emitting before trusting the parsed numbers

Thermal is PNG because at 256x192 greyscale a lossless frame is only ~40 KB;
compression buys little and costs detail the model needs. It stays greyscale
white-hot to match the training data for thermal_yolo11n_v3.

CO and CH4 from MEMS sensors are QUALITATIVE. DFRobot's own documentation
says they cannot produce calibrated ppm. Treat them as relative indices and
label them that way — a linear voltage-delta gives a number, not a
concentration.

Needs: pip3 install flask opencv-python pyserial

Run:    python3 rover_cams.py
View:   http://<RDK_IP>:8081
Stop:   Ctrl+C
"""

import glob
import json
import os
import re
import subprocess
import threading
import time
from collections import deque

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

try:
    import serial
except ImportError:
    serial = None

# ---------------------------------------------------------------- config

RGB_W, RGB_H, RGB_FOURCC = 640, 480, "NV12"
THM_W, THM_H, THM_FOURCC = 256, 192, "YUYV"

RGB_SENSOR_FPS = 30
THM_SENSOR_FPS = 50      # module offers 25 or 50 only; 40 cap applied below

THM_INVERT = False       # True if your module outputs black-hot
RGB_JPEG_Q = 60
PNG_LEVEL = 1            # 0-9; 1 is near-instant and still lossless
PORT = 8081

SERIAL_BAUD = 115200
SERIAL_STALE_S = 10      # no line for this long => mark stale

SETTINGS = {
    "rgb_max_fps": 30, "rgb_sharpness": 0.0,
    "thm_max_fps": 40, "thm_sharpness": 0.0,
}
SET_LOCK = threading.Lock()
LIMITS = {"rgb_max_fps": (1, 30), "thm_max_fps": (1, 40),
          "rgb_sharpness": (0.0, 2.0), "thm_sharpness": (0.0, 2.0)}

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_FILE = "Rover_Ground_Station.html"

# ---------------------------------------------------------------- detection


def probe(dev):
    try:
        r = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats-ext"],
                           capture_output=True, text=True, timeout=3)
        return r.stdout
    except Exception:
        return ""


def find_nodes():
    """Locate both cameras by format signature.

    Thermal : YUYV at 256x192 — distinctive, no other node offers it.
    RGB     : NV12 with Stepwise sizes — the scaler (VSE) output. The
              Discrete-only NV12 nodes are SIF/ISP stages that accept a
              format request then hang forever on VIDIOC_STREAMON.

    Numbering shifts between boots depending on whether the USB camera or
    the MIPI driver wins the race, so this must run every startup.
    """
    rgb = thermal = None
    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda p: int(p.rsplit("video", 1)[1]))
    for dev in nodes:
        out = probe(dev)
        if not out:
            continue
        if thermal is None and "YUYV" in out and f"{THM_W}x{THM_H}" in out:
            thermal = dev
        if rgb is None and "NV12" in out and "Stepwise" in out:
            rgb = dev
        if rgb and thermal:
            break
    return rgb, thermal


def sharpen(img, amount):
    """Unsharp mask. Cheap, and undone by setting the slider back to 0."""
    if amount <= 0.01:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)

# ---------------------------------------------------------------- gas

# Maps whatever the firmware calls a field onto the names the page expects.
# Add aliases here rather than changing firmware.
ALIASES = {
    "co": "co", "co_ppm": "co", "mems_co": "co", "carbonmonoxide": "co",
    "ch4": "ch4", "ch4_ppm": "ch4", "mems_ch4": "ch4", "methane": "ch4",
    "eco2": "eco2", "co2": "eco2", "eco2_ppm": "eco2",
    "tvoc": "tvoc", "voc": "tvoc", "tvoc_ppb": "tvoc",
    "aqi": "aqi", "iaq": "aqi",
    "temp": "temp", "temperature": "temp", "temp_c": "temp",
    "hum": "hum", "humidity": "hum", "hum_pct": "hum",
}

GAS = {"co": None, "ch4": None, "eco2": None, "tvoc": None,
       "aqi": None, "temp": None, "hum": None}
GAS_META = {"port": None, "connected": False, "last_rx": 0.0, "lines": 0}
GAS_LOCK = threading.Lock()
RAW = deque(maxlen=20)

KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(-?\d+(?:\.\d+)?)")
SEEN_JSON = False   # set once a "#J {...}" line arrives


def parse_line(line):
    """Prefer the '#J {...}' JSON line; fall back to loose key=value text.

    Once a JSON line has been seen, loose parsing is disabled permanently.
    Human-readable output is ambiguous to a parser: 'T/RH: 31.0C  80%' looks
    like the pair RH=31.0, which files temperature as humidity and drops the
    real humidity. Silently wrong beats loudly broken only for the machine.
    """
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

    if SEEN_JSON:
        return {}

    out = {}
    for k, v in KV.findall(line):
        key = ALIASES.get(k.lower())
        if key:
            out[key] = float(v)
    return out


def find_serial_port():
    ports = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
    return ports[0] if ports else None


class GasReader(threading.Thread):
    """Reads one JSON/key-value line per message from the ESP32."""

    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        if serial is None:
            print("[gas] pyserial not installed — pip3 install pyserial")
            return

        while True:
            port = find_serial_port()
            if not port:
                with GAS_LOCK:
                    GAS_META["connected"] = False
                    GAS_META["port"] = None
                time.sleep(3)
                continue

            try:
                ser = serial.Serial(port, SERIAL_BAUD, timeout=2)
                print(f"[gas] reading {port} @ {SERIAL_BAUD}")
                with GAS_LOCK:
                    GAS_META["port"] = port
                    GAS_META["connected"] = True
            except Exception as e:
                print(f"[gas] cannot open {port}: {e}")
                time.sleep(3)
                continue

            try:
                while True:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    vals = parse_line(line)
                    with GAS_LOCK:
                        RAW.append(line[:200])
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

# ---------------------------------------------------------------- capture


class Camera(threading.Thread):
    """Reads one device continuously, keeping only the newest encoded frame."""

    def __init__(self, name, dev, w, h, sensor_fps, fourcc,
                 fps_key, sharp_key, encoder, mime, post=None):
        super().__init__(daemon=True)
        self.name, self.dev, self.post = name, dev, post
        self.w, self.h, self.sensor_fps, self.fourcc = w, h, sensor_fps, fourcc
        self.fps_key, self.sharp_key = fps_key, sharp_key
        self.encoder, self.mime = encoder, mime
        self.frame = None
        self.count = 0
        self.fps_now = 0.0
        self.lock = threading.Lock()
        self.alive = False

    def open(self):
        cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.sensor_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        if self.dev is None:
            print(f"[{self.name}] no matching device found, not starting")
            return
        cap = self.open()
        if not cap.isOpened():
            print(f"[{self.name}] cannot open {self.dev} "
                  f"(is something else holding it?)")
            return

        print(f"[{self.name}] streaming from {self.dev} as {self.mime}")
        fails, t0, n0, last_emit = 0, time.time(), 0, 0.0

        while True:
            ok, f = cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    print(f"[{self.name}] read failing, reopening {self.dev}")
                    cap.release()
                    time.sleep(1)
                    cap = self.open()
                    fails = 0
                continue
            fails = 0
            self.alive = True

            with SET_LOCK:
                cap_fps = SETTINGS[self.fps_key]
                amount = SETTINGS[self.sharp_key]

            # Frames above the cap are read (keeping the buffer current) but
            # never encoded — encoding is where the CPU actually goes.
            now = time.time()
            if cap_fps > 0 and (now - last_emit) < (1.0 / cap_fps):
                continue
            last_emit = now

            if self.post:
                f = self.post(f)
            f = sharpen(f, amount)

            ok, buf = self.encoder(f)
            if ok:
                with self.lock:
                    self.frame = buf.tobytes()
                    self.count += 1

            dt = now - t0
            if dt >= 1.0:
                self.fps_now = round((self.count - n0) / dt, 1)
                t0, n0 = now, self.count


def enc_jpeg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, RGB_JPEG_Q])


def enc_png(img):
    return cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])


def thermal_post(f):
    """White-hot greyscale at native resolution — no colour map, no upscale."""
    grey = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_not(grey) if THM_INVERT else grey

# ---------------------------------------------------------------- server

app = Flask(__name__)


def stream(cam):
    last = -1
    header = b"--frame\r\nContent-Type: " + cam.mime.encode() + b"\r\n\r\n"
    while True:
        with cam.lock:
            n, buf = cam.count, cam.frame
        if buf is None or n == last:
            time.sleep(0.004)
            continue
        last = n
        yield header + buf + b"\r\n"


@app.route("/rgb")
def rgb_feed():
    return Response(stream(rgb),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/thermal")
def thermal_feed():
    return Response(stream(thermal),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        with SET_LOCK:
            for k, v in data.items():
                if k not in SETTINGS:
                    continue
                lo, hi = LIMITS[k]
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                if k.endswith("_max_fps"):
                    val = int(round(val))
                SETTINGS[k] = max(lo, min(hi, val))
    with SET_LOCK:
        return jsonify(dict(SETTINGS))


@app.route("/gas")
def gas():
    with GAS_LOCK:
        out = {k: v for k, v in GAS.items() if v is not None}
        age = time.time() - GAS_META["last_rx"] if GAS_META["last_rx"] else None
        out["_meta"] = {
            "port": GAS_META["port"],
            "connected": GAS_META["connected"],
            "lines": GAS_META["lines"],
            "age_s": round(age, 1) if age is not None else None,
            "stale": age is None or age > SERIAL_STALE_S,
        }
    return jsonify(out)


@app.route("/gas/raw")
def gas_raw():
    with GAS_LOCK:
        return jsonify({"port": GAS_META["port"], "lines": list(RAW)})


@app.route("/status")
def status():
    return {"rgb_dev": rgb.dev, "thermal_dev": thermal.dev,
            "rgb_alive": rgb.alive, "thermal_alive": thermal.alive,
            "rgb_fps": rgb.fps_now, "thermal_fps": thermal.fps_now,
            "rgb_frames": rgb.count, "thermal_frames": thermal.count}


FALLBACK = """<!doctype html><meta charset=utf-8>
<title>Rover cameras</title>
<style>body{background:#0B0B0D;color:#E8E8EA;font:13px ui-monospace,monospace;
padding:20px;margin:0}.g{display:grid;grid-template-columns:1fr 1fr;gap:12px}
img{width:100%;background:#000;border:1px solid rgba(255,255,255,.08);
border-radius:14px}p{color:rgba(232,232,234,.45)}</style>
<p>Rover_Ground_Station.html not found next to the script &mdash; plain view.
Gas JSON at <a href="/gas" style="color:#C6F24E">/gas</a>,
raw serial at <a href="/gas/raw" style="color:#C6F24E">/gas/raw</a>.</p>
<div class=g><img src="/rgb"><img src="/thermal"></div>"""


@app.route("/")
def index():
    if os.path.exists(os.path.join(HERE, PAGE_FILE)):
        return send_from_directory(HERE, PAGE_FILE)
    return FALLBACK


if __name__ == "__main__":
    print("probing video nodes...")
    RGB_DEV, THM_DEV = find_nodes()
    print(f"  RGB     -> {RGB_DEV or 'NOT FOUND'}")
    print(f"  THERMAL -> {THM_DEV or 'NOT FOUND'}")
    print(f"  SERIAL  -> {find_serial_port() or 'NOT FOUND'}")

    page = os.path.join(HERE, PAGE_FILE)
    print(f"  PAGE    -> {page if os.path.exists(page) else 'fallback (not found)'}")

    rgb = Camera("rgb", RGB_DEV, RGB_W, RGB_H, RGB_SENSOR_FPS, RGB_FOURCC,
                 "rgb_max_fps", "rgb_sharpness", enc_jpeg, "image/jpeg")
    thermal = Camera("thermal", THM_DEV, THM_W, THM_H, THM_SENSOR_FPS,
                     THM_FOURCC, "thm_max_fps", "thm_sharpness",
                     enc_png, "image/png", post=thermal_post)

    rgb.start()
    thermal.start()
    GasReader().start()
    time.sleep(1.5)
    print(f"\n  Open http://<RDK_IP>:{PORT}")
    print(f"  Check raw serial at http://<RDK_IP>:{PORT}/gas/raw\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
