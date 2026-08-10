#!/usr/bin/env python3
"""
sar_cams.py — serves both rover cameras as MJPEG streams on one page.

    RGB     : IMX219 NoIR on /dev/video8   (MIPI, NV12)
    Thermal : UVC module on /dev/video32   (USB, YUYV 256x192)

Run:    python3 sar_cams.py
View:   http://<RDK_IP>:8081
Stop:   Ctrl+C

Nothing else may hold the devices. Clear stragglers first:
    pkill -9 -f gst-launch
    sudo fuser -k 8081/tcp
"""

import threading
import time

import cv2
from flask import Flask, Response

# ---------------------------------------------------------------- config

RGB_DEV, RGB_W, RGB_H, RGB_FPS, RGB_FOURCC = 8, 640, 480, 30, "NV12"
THM_DEV, THM_W, THM_H, THM_FPS, THM_FOURCC = 32, 256, 192, 50, "YUYV"

THM_UPSCALE = (768, 576)            # 256x192 is unreadable at native size
THM_PALETTE = cv2.COLORMAP_INFERNO  # try JET or TURBO if you prefer
JPEG_Q = 70
PORT = 8081

# ---------------------------------------------------------------- capture


class Camera(threading.Thread):
    """Reads one device continuously, keeps only the newest frame.

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
        # Device path, not integer index — more reliable on a board with 30+ nodes.
        cap = cv2.VideoCapture(f"/dev/video{self.dev}", cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # newest frame, not a backlog
        return cap

    def run(self):
        cap = self.open()
        if not cap.isOpened():
            print(f"[{self.name}] cannot open /dev/video{self.dev} "
                  f"— something else is probably holding it")
            return

        fails = 0
        while True:
            ok, f = cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    print(f"[{self.name}] read failing, reopening")
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


rgb = Camera("rgb", RGB_DEV, RGB_W, RGB_H, RGB_FPS, RGB_FOURCC)
thermal = Camera("thermal", THM_DEV, THM_W, THM_H, THM_FPS, THM_FOURCC,
                 post=thermal_post)

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
    return {"rgb": rgb.alive, "thermal": thermal.alive,
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
        <span class="spec">IMX219 &middot; video8 &middot; 640&times;480</span></div>
      <img src="/rgb" alt="Visible-light camera feed">
    </section>

    <section class="cam heat">
      <div class="bar"><span class="dot"></span>Thermal
        <span class="spec">UVC &middot; video32 &middot; 256&times;192</span></div>
      <img src="/thermal" alt="Thermal camera feed">
    </section>
  </div>

  <footer>Both feeds are live MJPEG. Reload if one stalls.</footer>

<script>
  document.getElementById('host').textContent = location.host;
</script>
</body>
</html>"""


@app.route("/")
def index():
    return PAGE


if __name__ == "__main__":
    rgb.start()
    thermal.start()
    time.sleep(1.5)
    print(f"\n  Open http://<RDK_IP>:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
