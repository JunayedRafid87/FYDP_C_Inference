#!/usr/bin/env python3
"""
prepare_calibration_data.py
Converts calibration images (JPEG/PNG) to 640x640 preprocessed float32 / NV12 raw tensors
required for OpenExplorer Post-Training Quantization (PTQ).

For thermal images, applies CLAHE contrast enhancement matching the runtime pipeline.
"""

import os
import glob
import cv2
import numpy as np
import argparse

def bgr_to_nv12(image):
    """Converts a BGR image (H, W, 3) to NV12 format (H * 3/2, W)."""
    h, w = image.shape[:2]
    yuv420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    # Reorganize U and V planes into interleaved UV (NV12)
    y_size = h * w
    u_size = (h // 2) * (w // 2)
    y_plane = yuv420[:y_size].reshape((h, w))
    u_plane = yuv420[y_size:y_size + u_size]
    v_plane = yuv420[y_size + u_size:y_size + 2 * u_size]
    
    uv_interleaved = np.empty((h // 2, w), dtype=np.uint8)
    uv_interleaved[:, 0::2] = u_plane.reshape((h // 2, w // 2))
    uv_interleaved[:, 1::2] = v_plane.reshape((h // 2, w // 2))
    
    nv12 = np.vstack((y_plane, uv_interleaved))
    return nv12

def process_images(src_dir, dst_dir, is_thermal=False, count=100):
    os.makedirs(dst_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(src_dir, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_paths:
        print(f"[-] No images found in {src_dir}. Generating synthetic calibration set...")
        for i in range(count):
            img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
            nv12 = bgr_to_nv12(img)
            out_path = os.path.join(dst_dir, f"cal_{i:04d}.bin")
            nv12.tofile(out_path)
        print(f"[✓] Generated {count} synthetic NV12 calibration tensors in {dst_dir}")
        return

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if is_thermal else None
    
    saved = 0
    for idx, path in enumerate(image_paths[:count]):
        img = cv2.imread(path)
        if img is None:
            continue
        
        if is_thermal:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            enhanced = clahe.apply(gray)
            img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Letterbox / resize to 640x640
        resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        nv12 = bgr_to_nv12(resized)
        
        out_name = os.path.splitext(os.path.basename(path))[0] + ".bin"
        out_path = os.path.join(dst_dir, out_name)
        nv12.tofile(out_path)
        saved += 1
        
    print(f"[✓] Successfully processed {saved} calibration images to {dst_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare PTQ calibration tensors for Horizon OpenExplorer")
    parser.add_argument("--src", type=str, default="./raw_images", help="Path to input calibration images")
    parser.add_argument("--dst", type=str, default="./cal", help="Output directory for .bin calibration files")
    parser.add_argument("--thermal", action="store_true", help="Apply thermal CLAHE preprocessing")
    parser.add_argument("--count", type=int, default=100, help="Number of calibration frames (default: 100)")
    args = parser.parse_args()
    
    process_images(args.src, args.dst, is_thermal=args.thermal, count=args.count)

if __name__ == "__main__":
    main()
