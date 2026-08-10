#!/usr/bin/env python3
"""
rover_cams.py — serves both rover cameras as MJPEG streams on one page.

    RGB     : IMX219 NoIR, MIPI CSI, NV12
    Thermal : UVC module, USB, YUYV 256x192

Device nodes are DETECTED AT STARTUP, not hardcoded. On this board the
numbering shifts between boots depending on whether the USB camera or the
MIPI driver wins the race — the thermal cam has appeared as both
/dev/video32 and /dev/video0, and RGB as both /dev/video8 and /dev/video10.
Probing by format signature makes boot order irrelevant.

Run:    python3 rover_cams.py
View:   http://<RDK_IP>:8081
Stop:   Ctrl+C

If a device won't open, something else holds it:
    sudo ss -lptn 'sport = :8081'
    fuser -v /dev/video*
    pkill -9 -f gst-launch
"""

import glob
import subprocess
import threading
import time

import cv2
from flask import Flask, Response

# ---------------------------------------------------------------- config

RGB_W, RGB_H, RGB_FPS, RGB_FOURCC = 640, 480, 30, "NV12"
THM_W, THM_H, THM_FPS, THM_FOURCC = 256, 192, 50, "YUYV"

THM_UPSCALE = (768, 576)            # 256x192 is unreadable at native size
THM_PALETTE = cv2.COLORMAP_INFERNO  # or COLORMAP_JET / COLORMAP_TURBO
JPEG_Q = 70
PORT = 8081

# ---------------------------------------------------------------- detection


def probe(dev):
    """Return the format listing for one node, or '' if it can't be read."""
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
    """Reads one device continuously, keeping only the newest frame.

    Decoupling capture from delivery means a slow browser can't stall the
    device, and several viewers share one read instead of fighting for it.
    """

    def __init__(self, name, dev, w, h, fps, fourcc, post=None):
        super().__init__(daemon=True)
        self.name, self.dev, self.post = name, dev, post
        self.w, self.h, self.fps, self.fourcc = w, h, fps, fourcc
        self.frame = None
        self.count = 0
        self.lock = threading.Lock()
        self.alive = False

    def open(self):
        cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # newest frame, not a backlog
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
        fails = 0
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
            ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ok:
                with self.lock:
                    self.frame = jpg.tobytes()
                    self.count += 1


def thermal_post(f):
    grey = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    f = cv2.applyColorMap(grey, THM_PALETTE)
    return cv2.resize(f, THM_UPSCALE, interpolation=cv2.INTER_NEAREST)

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
            "rgb_frames": rgb.count, "thermal_frames": thermal.count}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rover cameras</title>
<style>
  :root{
    --ink:#0d0f0e; --panel:#161a18; --line:#2b322e;
    --text:#d8ddd6; --dim:#7c8a80; --live:#7fd18a; --heat:#ff7a3d;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--ink); color:var(--text);
    font:14px/1.45 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    padding:20px;
  }
  header{
    display:flex; align-items:baseline; gap:14px;
    border-bottom:1px solid var(--line); padding-bottom:12px; margin-bottom:20px;
  }
  h1{font-size:15px; font-weight:600; margin:0; letter-spacing:.14em; text-transform:uppercase}
  .host{color:var(--dim); font-size:12px}
  .grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:18px}
  .cam{background:var(--panel); border:1px solid var(--line); border-radius:3px; overflow:hidden}
  .bar{
    display:flex; align-items:center; gap:9px;
    padding:9px 12px; border-bottom:1px solid var(--line);
    font-size:11px; letter-spacing:.16em; text-transform:uppercase;
  }
  .dot{width:7px; height:7px; border-radius:50%; background:var(--live)}
  .cam.heat .dot{background:var(--heat)}
  .spec{margin-left:auto; color:var(--dim); letter-spacing:.04em; text-transform:none}
  img{display:block; width:100%; height:auto; background:#000}
  footer{margin-top:20px; color:var(--dim); font-size:12px}
  @media (prefers-reduced-motion:no-preference){
    .dot{animation:pulse 2.4s ease-in-out infinite}
    @keyframes pulse{50%{opacity:.35}}
  }
</style>
</head>
<body>
  <header>
    <h1>Rover cameras</h1>
    <span class="host" id="host"></span>
  </header>

  <div class="grid">
    <section class="cam">
      <div class="bar"><span class="dot"></span>Visible / NoIR
        <span class="spec" id="rgbdev">IMX219</span></div>
      <img src="/rgb" alt="Visible-light camera feed">
    </section>

    <section class="cam heat">
      <div class="bar"><span class="dot"></span>Thermal
        <span class="spec" id="thmdev">UVC</span></div>
      <img src="/thermal" alt="Thermal camera feed">
    </section>
  </div>

  <footer>Live MJPEG. Reload if a feed stalls. Device map at <a href="/status" style="color:var(--dim)">/status</a>.</footer>

<script>
  document.getElementById('host').textContent = location.host;
  fetch('/status').then(r => r.json()).then(s => {
    document.getElementById('rgbdev').textContent = 'IMX219 \\u00b7 ' + (s.rgb_dev || 'not found');
    document.getElementById('thmdev').textContent = 'UVC \\u00b7 ' + (s.thermal_dev || 'not found');
  }).catch(() => {});
</script>
</body>
</html>"""


@app.route("/")
def index():
    return PAGE


if __name__ == "__main__":
    print("probing video nodes...")
    RGB_DEV, THM_DEV = find_nodes()
    print(f"  RGB     -> {RGB_DEV or 'NOT FOUND'}")
    print(f"  THERMAL -> {THM_DEV or 'NOT FOUND'}")

    rgb = Camera("rgb", RGB_DEV, RGB_W, RGB_H, RGB_FPS, RGB_FOURCC)
    thermal = Camera("thermal", THM_DEV, THM_W, THM_H, THM_FPS, THM_FOURCC,
                     post=thermal_post)

    rgb.start()
    thermal.start()
    time.sleep(1.5)
    print(f"\n  Open http://<RDK_IP>:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
