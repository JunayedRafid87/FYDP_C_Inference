#!/usr/bin/env python3
"""
infer_test.py — one-shot CPU inference check on a live camera frame.

Grabs a single frame, runs a YOLO11 ONNX model on it via ONNX Runtime,
draws boxes, writes an annotated JPEG, and reports timing.

The point is not speed — it's finding out whether the model works on YOUR
sensor. Both models were trained on public thermal/RGB datasets, none of
which used this camera. Held-out mAP does not measure that gap; this does.

Usage:
    python3 infer_test.py <model.onnx> <device> [conf]

    python3 infer_test.py thermal_yolo11n_v3.onnx /dev/video0
    python3 infer_test.py rgb_yolo11n_best.onnx /dev/video10 0.35

Device numbering shifts between boots — check with `v4l2-ctl --list-devices`
first. Nothing else may hold the camera:
    pkill -9 -f rover_cams.py

Needs: pip3 install onnxruntime
"""

import sys
import time

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime missing — pip3 install onnxruntime")

IOU = 0.5
OUT = "/home/sunrise/detect.jpg"
RAW = "/home/sunrise/detect_input.jpg"


def letterbox(img, size):
    """Resize preserving aspect ratio, pad to square with grey.

    Must match the training preprocessing. Ultralytics letterboxes with 114
    grey by default; a plain resize would distort aspect ratio and shift
    every box.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return out, r, left, top


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    model, dev = sys.argv[1], sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
    print(f"model   {model}")
    print(f"  input  {inp.name} {inp.shape}")
    for o in sess.get_outputs():
        print(f"  output {o.name} {o.shape}")

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"cannot open {dev} — is something else holding it? "
                 f"try: fuser -v {dev}")

    # First frames after stream start are often exposure garbage.
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        sys.exit(f"could not read a frame from {dev}")

    print(f"frame   {frame.shape} from {dev}")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(RAW, frame)

    img, r, dx, dy = letterbox(frame, size)
    blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # Run twice: the first call includes lazy graph setup and is not
    # representative of steady-state throughput.
    sess.run(None, {inp.name: blob})
    t0 = time.time()
    out = sess.run(None, {inp.name: blob})[0]
    dt = time.time() - t0

    p = out[0].T                     # [1,5,8400] -> [8400,5]
    scores = p[:, 4]
    keep = scores > conf
    p, scores = p[keep], scores[keep]

    print(f"\ninference {dt * 1000:.0f} ms  ({1 / dt:.1f} fps if sustained)")
    print(f"raw max score {out[0][4].max():.3f}")
    print(f"{len(p)} boxes above conf {conf}")

    n = 0
    if len(p):
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([(cx - w / 2 - dx) / r,
                          (cy - h / 2 - dy) / r,
                          w / r, h / r], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, IOU)
        for i in np.array(idx).flatten():
            x, y, bw, bh = boxes[i].astype(int)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"{scores[i]:.2f}", (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            n += 1
        print(f"{n} after NMS")

    cv2.imwrite(OUT, frame)
    print(f"\nwrote {OUT}  (raw frame at {RAW})")

    if n == 0:
        print("\nNo detections. Before blaming the model, check:")
        print(f"  - open {RAW} — is the frame actually showing a person?")
        print("  - 'raw max score' above: near 0 means the model sees nothing,")
        print("    0.2-0.4 means it half-sees and the threshold is too high")
        print("  - preprocessing mismatch (CLAHE? normalisation?) vs training")


if __name__ == "__main__":
    main()
