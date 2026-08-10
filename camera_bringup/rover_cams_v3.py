#!/usr/bin/env python3
"""
rover_cams.py — ground station page with both live camera feeds.

    RGB     : IMX219 NoIR, MIPI CSI, NV12
    Thermal : UVC module, USB, YUYV 256x192, WHITE-HOT greyscale

Thermal is deliberately left as greyscale white-hot, matching the training
data for thermal_yolo11n_v3. Applying a false-colour palette before
inference would feed the model colour relationships it never learned.

Device nodes are DETECTED AT STARTUP, not hardcoded. Numbering on this
board shifts between boots depending on whether the USB camera or the MIPI
driver wins the race — thermal has appeared as both /dev/video32 and
/dev/video0, RGB as both /dev/video8 and /dev/video10.

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

THM_UPSCALE = (768, 576)   # 256x192 is unreadable at native size
THM_INVERT = False         # set True if your module outputs black-hot
JPEG_Q = 70
PORT = 8081

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
        self.fps_now = 0.0
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
        fails, t0, n0 = 0, time.time(), 0

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

            dt = time.time() - t0
            if dt >= 1.0:
                self.fps_now = round((self.count - n0) / dt, 1)
                t0, n0 = time.time(), self.count


def thermal_post(f):
    """White-hot greyscale — no false colour.

    The model was trained on white-hot imagery, so the inference frame must
    match. Encoding single-channel keeps the JPEG smaller too.
    """
    grey = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if THM_INVERT:
        grey = cv2.bitwise_not(grey)
    return cv2.resize(grey, THM_UPSCALE, interpolation=cv2.INTER_NEAREST)

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


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rover Ground Station</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0B0B0D; --well:#08080A;
    --text:#E8E8EA; --dim:rgba(232,232,234,.35); --dimmer:rgba(232,232,234,.55);
    --edge:rgba(255,255,255,.08);
    --orange:#FF5A36; --lime:#C6F24E;
  }
  *{margin:0; padding:0; box-sizing:border-box}
  body{
    background:var(--bg); color:var(--text);
    font-family:'Roboto Mono',ui-monospace,monospace;
    min-height:100vh; padding:20px;
    display:flex; flex-direction:column; gap:16px;
  }
  .head{display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap}
  .title{display:flex; align-items:baseline; gap:12px}
  .title h1{font-size:12px; font-weight:600; letter-spacing:.2em}
  .sub{font-size:10.5px; letter-spacing:.08em; color:var(--dim)}
  .tracks{font-size:10.5px; color:var(--orange)}

  .grid{
    flex:1 1 auto; min-height:0;
    display:grid; grid-template-columns:1fr 1fr; gap:12px;
  }
  @media (max-width:820px){ .grid{grid-template-columns:1fr} }

  .cam{
    position:relative; min-height:340px;
    border-radius:14px; overflow:hidden;
    background:var(--well); border:1px solid var(--edge);
  }
  .cam img{width:100%; height:100%; object-fit:contain; display:block; background:#000}

  .tag{
    position:absolute; top:10px; left:12px;
    display:flex; align-items:center; gap:8px;
    font-size:10px; letter-spacing:.1em; color:rgba(255,255,255,.8);
  }
  .dot{width:6px; height:6px; border-radius:50%; background:var(--orange)}
  .heat .dot{background:var(--lime)}

  .meta{
    position:absolute; bottom:10px; left:12px; right:12px;
    display:flex; justify-content:space-between;
    font-size:10px; color:var(--dimmer);
  }
  .ramp{
    position:absolute; top:10px; right:12px; width:10px; bottom:34px;
    border-radius:4px; border:1px solid rgba(255,255,255,.2);
    background:linear-gradient(180deg,#FFFFFF,#000000);
  }
  .foot{font-size:10.5px; color:var(--dim); letter-spacing:.06em}
  .foot a{color:var(--dimmer)}
  @media (prefers-reduced-motion:no-preference){
    .dot{animation:blink 1.6s ease-in-out infinite}
    @keyframes blink{50%{opacity:.25}}
  }
</style>
</head>
<body>

  <div class="head">
    <div class="title">
      <h1>VISION</h1>
      <span class="sub">IMX219 NoIR &middot; UVC LWIR 256&times;192</span>
    </div>
    <span class="tracks" id="host"></span>
  </div>

  <div class="grid">
    <section class="cam">
      <img src="/rgb" alt="Visible-light camera feed">
      <div class="tag"><span class="dot"></span><span id="rgbdev">RGB</span></div>
      <div class="meta">
        <span>NV12 640&times;480</span>
        <span id="rgbfps">&mdash; fps</span>
      </div>
    </section>

    <section class="cam heat">
      <img src="/thermal" alt="Thermal camera feed, white-hot">
      <div class="tag"><span class="dot"></span><span id="thmdev">THERMAL</span></div>
      <div class="ramp" title="white-hot"></div>
      <div class="meta">
        <span>YUYV 256&times;192 &middot; WHITE-HOT</span>
        <span id="thmfps">&mdash; fps</span>
      </div>
    </section>
  </div>

  <div class="foot">Live MJPEG &middot; device map at <a href="/status">/status</a></div>

<script>
  document.getElementById('host').textContent = location.host;
  function tick(){
    fetch('/status').then(r => r.json()).then(s => {
      document.getElementById('rgbdev').textContent =
        'RGB \\u00b7 ' + (s.rgb_dev || 'NOT FOUND');
      document.getElementById('thmdev').textContent =
        'THERMAL \\u00b7 ' + (s.thermal_dev || 'NOT FOUND');
      document.getElementById('rgbfps').textContent = s.rgb_fps + ' fps';
      document.getElementById('thmfps').textContent = s.thermal_fps + ' fps';
    }).catch(() => {});
  }
  tick();
  setInterval(tick, 2000);
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
