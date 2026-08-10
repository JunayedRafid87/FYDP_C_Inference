#!/usr/bin/env python3
"""
rover_cams.py — serves the ground station page and both live camera feeds.

    RGB     : IMX219 NoIR, MIPI CSI, NV12
    Thermal : UVC module, USB, YUYV 256x192, WHITE-HOT greyscale

Put Rover_Ground_Station.html next to this script and it gets served at /.
Serving the page from here (rather than opening it as a local file) keeps
the feeds same-origin — absolute http:// URLs get silently upgraded to
https:// by the browser when the page loads from an https context, and the
resulting TLS handshake shows up in the log as 'Bad request syntax'.

Thermal stays greyscale white-hot to match the training data for
thermal_yolo11n_v3, and is sent at NATIVE 256x192 with high JPEG quality —
upscaling server-side cost CPU and made it look worse, since the browser
scales it anyway.

Run:    python3 rover_cams.py
View:   http://<RDK_IP>:8081
Stop:   Ctrl+C
"""

import glob
import os
import subprocess
import threading
import time

import cv2
from flask import Flask, Response, send_from_directory

# ---------------------------------------------------------------- config

RGB_W, RGB_H, RGB_FPS, RGB_FOURCC = 640, 480, 30, "NV12"
THM_W, THM_H, THM_FPS, THM_FOURCC = 256, 192, 50, "YUYV"

THM_INVERT = False    # True if your module outputs black-hot
RGB_JPEG_Q = 60       # main CPU lever — drop to 45 if fps still sags
THM_JPEG_Q = 90       # tiny frames, so quality is cheap here
PORT = 8081

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

# ---------------------------------------------------------------- capture


class Camera(threading.Thread):
    """Reads one device continuously, keeping only the newest frame."""

    def __init__(self, name, dev, w, h, fps, fourcc, quality, post=None):
        super().__init__(daemon=True)
        self.name, self.dev, self.post = name, dev, post
        self.w, self.h, self.fps, self.fourcc = w, h, fps, fourcc
        self.quality = quality
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
        cap.set(cv2.CAP_PROP_FPS, self.fps)
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

        print(f"[{self.name}] streaming from {self.dev}")
        fails, t0, n0 = 0, time.time(), 0
        enc = [cv2.IMWRITE_JPEG_QUALITY, self.quality]

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
            if self.post:
                f = self.post(f)
            ok, jpg = cv2.imencode(".jpg", f, enc)
            if ok:
                with self.lock:
                    self.frame = jpg.tobytes()
                    self.count += 1

            dt = time.time() - t0
            if dt >= 1.0:
                self.fps_now = round((self.count - n0) / dt, 1)
                t0, n0 = time.time(), self.count


def thermal_post(f):
    """White-hot greyscale at native resolution.

    No colour map: the model was trained on white-hot, and false colour
    would feed it relationships it never learned. No upscale either — the
    browser does that for free and does it better than INTER_NEAREST.
    """
    grey = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_not(grey) if THM_INVERT else grey

# ---------------------------------------------------------------- server

app = Flask(__name__)


def mjpeg(cam):
    last = -1
    while True:
        with cam.lock:
            n, buf = cam.count, cam.frame
        if buf is None or n == last:
            time.sleep(0.005)
            continue
        last = n
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")


@app.route("/rgb")
def rgb_feed():
    return Response(mjpeg(rgb),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/thermal")
def thermal_feed():
    return Response(mjpeg(thermal),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


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
<p>Rover_Ground_Station.html not found next to the script &mdash; plain view.</p>
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

    page = os.path.join(HERE, PAGE_FILE)
    print(f"  PAGE    -> {page if os.path.exists(page) else 'fallback (file not found)'}")

    rgb = Camera("rgb", RGB_DEV, RGB_W, RGB_H, RGB_FPS, RGB_FOURCC, RGB_JPEG_Q)
    thermal = Camera("thermal", THM_DEV, THM_W, THM_H, THM_FPS, THM_FOURCC,
                     THM_JPEG_Q, post=thermal_post)

    rgb.start()
    thermal.start()
    time.sleep(1.5)
    print(f"\n  Open http://<RDK_IP>:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
