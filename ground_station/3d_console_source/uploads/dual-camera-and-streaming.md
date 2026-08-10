# Dual camera setup — RDK X5

Getting an IMX219 NoIR module (MIPI) and a 256×192 UVC thermal module (USB) working together on a D-Robotics RDK X5 4GB, running RDK OS / Ubuntu 22.04, kernel 6.1.83.

**Short version: RGB is `/dev/video8`. Thermal is `/dev/video32`. Skip `mipi_cam`, `hobot_codec`, and the `/app/pydev_demo` MIPI samples — none of them work on this firmware image.**

---

## Hardware

| Item | Detail |
|---|---|
| Board | RDK X5 4GB |
| RGB sensor | Raspberry Pi NoIR Camera Module v2 (Sony IMX219, 8MP), MIPI CSI |
| Thermal sensor | 256×192 UVC module (USB), YUYV, 25/50 fps |
| Cable | The genuine Pi NoIR v2 board is **15-pin**; the RDK X5 is **22-pin**. You need a 15→22 pin cable (Pi Zero / Pi 5 style). |

**Never connect or disconnect the MIPI camera while the board is powered.** The vendor docs warn it can damage the module. USB is hot-pluggable as normal.

FFC contacts face away from the black latch — blue backing up.

---

## Why the vendor tooling fails

This image ships **two incompatible camera stacks**, and most of the documentation describes the one that isn't there.

| Stack | Device nodes | Status |
|---|---|---|
| Hobot VIO | `/dev/mipi*`, `/dev/vps*`, `/dev/isp*` | **Absent.** `mipi_cam` and the `pydev_demo` samples target these and always fail. |
| V4L2 media-controller | `/dev/video0`–`/dev/video31`, `/dev/media0` | **The live one.** Driver is `vs-video (platform:vscam)`. |

Check for yourself:

```bash
ls /dev | grep -i -E "mipi|vin|vps|isp|vio"   # prints nothing
v4l2-ctl --list-devices                        # vs-video, 32 nodes
```

`mipi_cam` reporting `There are no available host.` means exactly this — it's looking for device nodes that don't exist. Not a cable fault, not a resource conflict, not something you did wrong.

The knock-on effects are all symptoms of the same root cause: `hobot_codec` reports `has not received image for more than 5 seconds` on `/hbmem_img`, and `websocket` reports `did not receive image data` on `/image_jpeg`.

A UVC camera enumerates separately with its own `/dev/media1` and never touches `vs-video`.

---

## RGB camera (IMX219, MIPI)

### Verify the sensor is bound

```bash
sudo media-ctl -d /dev/media0 -p 2>/dev/null | grep -A3 imx219
```

Expect `imx219 4-0010` or `imx219 6-0010`. Either is fine.

Confirm it has a real format:

```bash
sudo media-ctl -d /dev/media0 --get-v4l2 '"imx219 6-0010":0'
```

Should report `SRGGB10_1X10/3264x2464@100/2100`. Zero or garbage means the sensor never got configured.

### Use the node that reports Stepwise sizes

```bash
sudo v4l2-ctl --device=/dev/video8 --list-formats-ext
```

```
[0]: 'NV12' (Y/UV 4:2:0)
    Size: Stepwise 64x64 - 1920x1080 with step 2/2
```

**`Stepwise` is the marker of the working node.** It's a scaler (VSE) output that accepts an arbitrary requested size, so `--set-fmt-video` takes effect. The Discrete-only nodes (`video0`, `video4`, `video6`) ignore format requests and then hang forever on `VIDIOC_STREAMON`.

### Test capture

```bash
sudo v4l2-ctl --device=/dev/video8 \
  --set-fmt-video=width=1280,height=720,pixelformat=NV12 \
  --stream-mmap --stream-count=10 --stream-to=test.raw
ls -la test.raw
```

Expect **13,824,000 bytes** — 10 × 1280 × 720 × 1.5.

---

## Thermal camera (UVC, USB)

Plug it in. No CSI ports, no ISP, no `cam-service`.

```bash
v4l2-ctl --list-devices
```

```
Camera: Camera (usb-xhci-hcd.2.auto-1.2):
	/dev/video32
	/dev/video33
	/dev/media1
```

No numbering collision — `vs-video` occupies video0–31, so USB cameras start at video32.

```bash
v4l2-ctl -d /dev/video32 --list-formats-ext
```

```
[0]: 'YUYV' (YUYV 4:2:2)
    Size: Discrete 256x192
        Interval: Discrete 0.020s (50.000 fps)
        Interval: Discrete 0.040s (25.000 fps)
```

**Check the height.** Many modules in this class output **256×384** — top half pseudo-colour, bottom half raw radiometric data. Split the frame before feeding a detection model if yours does. This one doesn't.

**Only 25 and 50 fps exist.** Asking for 30 negotiates down to 25.

```bash
timeout 15 v4l2-ctl -d /dev/video32 \
  --set-fmt-video=width=256,height=192,pixelformat=YUYV \
  --stream-mmap --stream-count=10 --stream-to=$HOME/thermal.raw
ls -la $HOME/thermal.raw
```

Expect **983,040 bytes** — 10 × 256 × 192 × 2.

Decode one frame:

```bash
ffmpeg -f rawvideo -pix_fmt yuyv422 -s 256x192 -i $HOME/thermal.raw -frames:v 1 thermal.png
```

---

## Option A — web page serving both cameras

A Flask MJPEG server. Preferred over GStreamer for two reasons: only a browser is needed on the viewing machine, and it's the same OpenCV code path that detection models plug into later.

```bash
pip3 install flask opencv-python
```

Run `rover_cams.py` (in this repo), then open `http://<RDK_IP>:8081`.

Design notes:

- One capture thread per camera, each keeping only the newest frame. A slow browser can't stall the device, and multiple viewers share one read.
- `CAP_PROP_BUFFERSIZE = 1` prevents latency creeping up as frames queue.
- `/status` returns JSON frame counts — use it to tell a stalled camera from a stalled browser.

**If the RGB feed fails but thermal works**, your OpenCV build can't do NV12 over V4L2. Check for GStreamer support:

```bash
python3 -c "import cv2; print([l for l in cv2.getBuildInformation().split('\n') if 'GStreamer' in l])"
```

If `YES`, replace the `cv2.VideoCapture(...)` call with a pipeline:

```python
cap = cv2.VideoCapture(
    f"v4l2src device=/dev/video{self.dev} ! "
    f"video/x-raw,format=NV12,width={self.w},height={self.h},framerate={self.fps}/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1",
    cv2.CAP_GSTREAMER)
```

If `NO`, `sudo apt install python3-opencv` gives you a build that has it.

Passing a device path (`f"/dev/video{n}"`) is more reliable than an integer index on a board with 32 nodes.

---

## Option B — GStreamer UDP streams

Lower latency, but needs a receiver running on the viewing machine.

```bash
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
ip addr show | grep "inet 192.168"
```

Four terminals: two receivers on the laptop, two senders on the RDK. **Receivers first** — UDP is fire-and-forget, so a sender with no listener discards packets silently. Each stream needs its own SSH session; pasting a second command into a terminal running the first will Ctrl-C the first.

**Laptop, RGB (5000):**
```bash
gst-launch-1.0 udpsrc port=5000 buffer-size=2097152 \
  ! application/x-rtp,encoding-name=JPEG,payload=26 \
  ! rtpjpegdepay ! queue ! jpegdec ! queue ! videoconvert \
  ! fpsdisplaysink video-sink=autovideosink sync=false text-overlay=false
```

**Laptop, thermal (5001):**
```bash
gst-launch-1.0 udpsrc port=5001 buffer-size=2097152 \
  ! application/x-rtp,encoding-name=JPEG,payload=26 \
  ! rtpjpegdepay ! queue ! jpegdec ! queue \
  ! videoscale ! video/x-raw,width=768,height=576 ! videoconvert \
  ! fpsdisplaysink video-sink=autovideosink sync=false text-overlay=false
```

**RDK, RGB:**
```bash
gst-launch-1.0 v4l2src device=/dev/video8 \
  ! video/x-raw,format=NV12,width=640,height=480,framerate=30/1 \
  ! queue max-size-buffers=2 leaky=downstream \
  ! videoconvert ! jpegenc quality=50 ! rtpjpegpay \
  ! udpsink host=<LAPTOP_IP> port=5000 sync=false
```

**RDK, thermal:**
```bash
gst-launch-1.0 v4l2src device=/dev/video32 \
  ! video/x-raw,format=YUY2,width=256,height=192,framerate=50/1 \
  ! queue max-size-buffers=2 leaky=downstream \
  ! videoconvert ! jpegenc quality=70 ! rtpjpegpay \
  ! udpsink host=<LAPTOP_IP> port=5001 sync=false
```

Swap `fpsdisplaysink ...` for plain `autovideosink` once tuning is done.

### GStreamer gotchas

- **`YUYV` in V4L2 is `YUY2` in GStreamer caps.** Using `YUYV` gives `could not link v4l2src0 to videoconvert0`.
- **Without `queue` elements the pipeline runs single-threaded** and blocks on the slowest stage. Symptom: `Pipeline construction is invalid, please add queues` plus `Not enough buffering available for the processing deadline`.
- **`leaky=downstream` drops stale frames** rather than queueing them. Without it latency grows without bound.
- **`jpegenc` is software and single-threaded.** One pegged core on an 8-core board shows as ~11% average load — read *per-core* usage in htop, not the average. Lower `quality` or resolution to buy frames.
- **The sender outlives a dead receiver.** If the window vanishes, check `ps -ef | grep gst-launch` on the RDK before restarting anything; usually only the laptop side needs relaunching.
- `nohup ... &` keeps a sender alive across SSH disconnects.

---

## Troubleshooting

### Device or port already in use

The single most common failure once you've been experimenting. OpenCV reports a busy device unhelpfully as `can't open camera by index`.

Find what holds a port:

```bash
sudo ss -lptn 'sport = :8081'
```

Find what holds a device:

```bash
fuser -v /dev/video8
```

Clear everything:

```bash
sudo pkill -9 -f rover_cams.py
sudo pkill -9 -f gst-launch
sudo fuser -k 8081/tcp
```

**`sudo pkill` can miss processes owned by your own user** depending on the pattern. If a process survives, kill it by PID — and use `-9` if plain `kill` is ignored:

```bash
kill -9 <PID>
```

Verify before relaunching. Both should print nothing:

```bash
sudo ss -lptn 'sport = :8081'
fuser -v /dev/video32
```

### Capture hangs forever with no output

You're on a Discrete-only node. Use `/dev/video8`.

Always bound exploratory captures so a hang can't eat your evening:

```bash
timeout 15 v4l2-ctl -d /dev/videoN --stream-mmap --stream-count=1 --stream-to=$HOME/f.raw
echo "exit: $?"   # 124 = timed out
```

### Decoded frame is uniformly green

In NV12 the neutral chroma value is 128; an all-zero UV plane decodes as pure green. The ISP isn't processing. Capturing from `/dev/video8` with an explicit format avoids it.

### ffmpeg complains about packet size

The requested resolution didn't take, so you're decoding at the wrong size:

```bash
v4l2-ctl -d /dev/video8 --get-fmt-video
```

| Resolution | NV12 bytes/frame | YUYV bytes/frame |
|---|---|---|
| 3264×2464 | 12,063,744 | — |
| 1920×1080 | 3,110,400 | — |
| 1280×720 | 1,382,400 | — |
| 256×192 | — | 98,304 |

### `scp` as root gets Permission denied

Root SSH login is disabled by default on the RDK image — the password was never the problem. Copy as `sunrise`:

```bash
scp sunrise@<RDK_IP>:/home/sunrise/frame.png .
```

### GPIO export fails with `Device or resource busy`

Already exported, usually from an earlier manual probe:

```bash
echo 351 > /sys/class/gpio/unexport
echo 353 > /sys/class/gpio/unexport
```

Leaving these exported can stop the driver resetting the sensor later.

### Driver unbind/rebind fails with `-16`

```bash
echo 6-0010 > /sys/bus/i2c/drivers/imx219/unbind   # Device or resource busy
```

The sensor's media links are `IMMUTABLE` and held by the pipeline. The unbind fails, but a following bind still runs and collides:

```
entity vs-snps-csi0-0 pad 1 <-> entity vs-sif0-0 pad 0 existed!
imx219 6-0010: failed to register sensor sub-device: -16
```

This leaves the sensor worse off than before. **Don't use unbind/rebind on this image.** A full power cycle recovers it.

---

## Pipeline topology

From `media-ctl -d /dev/media0 -p`. Each CSI port fans out through a SIF (raw), an ISP, and a VSE scaler:

```
imx219 → vs-snps-csiN-0 → vs-sifN-0 ─┬→ /dev/video{0,1,2,3}   raw Bayer
                                      └→ vs-isp0-N ─┬→ /dev/video{4,5,6,7}   ISP out
                                                     └→ vs-vse0-N → /dev/video{8..31}  scaler out
```

| CSI | SIF node | ISP node | Scaler nodes |
|---|---|---|---|
| 0 | video0 | video4 | video8–13 |
| 1 | video1 | video5 | video14–19 |
| 2 | video2 | video6 | video20–25 |
| 3 | video3 | video7 | video26–31 |

32 nodes does **not** mean 32 cameras — each is one stage of the pipeline.

---

## Open questions

- **Bus numbering moves between boots.** The sensor has appeared as both `imx219 4-0010` (CSI2) and `imx219 6-0010` (CSI0), and did not reliably track which physical connector was used. Check with `media-ctl` each session rather than assuming.
- **`/dev/video8` worked regardless of which bus the sensor reported.** The scaler branch appears more forgiving than the SIF and ISP branches, but the mechanism isn't confirmed.
- **`cam-service` runs at boot as `-C5 3 5 3 -s4 2 4 2 -i6 -V6`.** Flag meanings undocumented; changing to `-i0 -V0` produced `VIDIOC_STREAMON returned -1` rather than a fix. Left at stock.
- **Camera calibration is unset.** `get camera calibration parameters failed` is harmless for plain capture but must be fixed before SLAM or depth work.
- **Hardware JPEG encoding unexplored.** The board exposes a `jpu` group, suggesting a hardware encoder. Using it instead of software `jpegenc` would free the saturated core.

---

## NoIR notes

- **NoIR is not thermal.** Removing the IR-cut filter extends sensitivity into near-infrared (~700–1000 nm). It sees in darkness *only with an IR illuminator* (850 nm LEDs or similar). No body heat, no temperature data — that's what the separate LWIR module is for.
- **Daylight images look washed-out and pinkish-magenta.** Correct NoIR behaviour, not a white-balance fault.

## BPU note

Models don't run on the X5's BPU as-is. A `.pt` file must go through the D-Robotics toolchain to become a `.bin` before you get hardware acceleration — otherwise it silently falls back to CPU and crawls. Budget time for the conversion.
