#!/usr/bin/env python3
"""
One-time patch: stops the thermal camera path from computing
grey+CLAHE equalization TWICE per frame (once for inference, once
for display). Now it's computed once and shared.

Usage:
    cd ~/FYDP_Test
    python3 apply_clahe_fix.py

It edits rdk_x5_stream_ground_v10.py in place, after saving a .bak backup.
Safe to re-run — it will just say "already applied" the second time.
"""

TARGET = "rdk_x5_stream_ground_v10.py"

OLD1 = '''def thermal_for_inference(frame):
    """CLAHE'd greyscale at NATIVE resolution, 3-channel.

    Matches the white-hot training data. No upscale, no colormap — both
    were costing CPU and moving the input away from what the model saw
    during training.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.cvtColor(CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)


def thermal_for_display(frame, colormap_type="inferno", target_w=640,
                        target_h=480, interp=cv2.INTER_LINEAR, sharpen=0.0):
    """The pretty version: CLAHE, bicubic upscale, false colour. Display only."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    up = cv2.resize(CLAHE.apply(gray), (target_w, target_h),
                    interpolation=interp)'''

NEW1 = '''def thermal_equalize(frame):
    """Grey + CLAHE only. Shared by inference and display paths below so
    this (identical) contrast-equalization work isn't done twice per frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return CLAHE.apply(gray)


def thermal_for_inference(frame, eq=None):
    """CLAHE'd greyscale at NATIVE resolution, 3-channel.

    Matches the white-hot training data. No upscale, no colormap — both
    were costing CPU and moving the input away from what the model saw
    during training.
    """
    eq = thermal_equalize(frame) if eq is None else eq
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def thermal_for_display(frame, colormap_type="inferno", target_w=640,
                        target_h=480, interp=cv2.INTER_LINEAR, sharpen=0.0,
                        eq=None):
    """The pretty version: CLAHE, bicubic upscale, false colour. Display only."""
    eq = thermal_equalize(frame) if eq is None else eq
    up = cv2.resize(eq, (target_w, target_h),
                    interpolation=interp)'''

OLD2 = '''        if self.is_thermal:
                ow, oh = self.out_size or (640, 480)
                infer_frame = (thermal_for_display(frame, self.colormap, ow, oh,
                                                   self.interp, self.sharpen)
                               if self.infer_on == "display"
                               else thermal_for_inference(frame))
                display = thermal_for_display(frame, self.colormap, ow, oh,
                                              self.interp, self.sharpen)
                sx = display.shape[1] / infer_frame.shape[1]
                sy = display.shape[0] / infer_frame.shape[0]'''

NEW2 = '''        if self.is_thermal:
                ow, oh = self.out_size or (640, 480)
                eq = thermal_equalize(frame)
                infer_frame = (thermal_for_display(frame, self.colormap, ow, oh,
                                                   self.interp, self.sharpen, eq=eq)
                               if self.infer_on == "display"
                               else thermal_for_inference(frame, eq=eq))
                display = thermal_for_display(frame, self.colormap, ow, oh,
                                              self.interp, self.sharpen, eq=eq)
                sx = display.shape[1] / infer_frame.shape[1]
                sy = display.shape[0] / infer_frame.shape[0]'''


def main():
    with open(TARGET) as f:
        content = f.read()

    if NEW1 in content and NEW2 in content:
        print(f"[skip] {TARGET} already has this fix applied. Nothing to do.")
        return

    if OLD1 not in content or OLD2 not in content:
        print(f"[error] Could not find the expected original code in {TARGET}.")
        print("        The file may have changed since this patch was written.")
        print("        No changes made. Send Claude the current file to get an updated patch.")
        return

    backup = TARGET + ".bak"
    with open(backup, "w") as f:
        f.write(content)
    print(f"[ok] Backup saved to {backup}")

    content = content.replace(OLD1, NEW1)
    content = content.replace(OLD2, NEW2)

    with open(TARGET, "w") as f:
        f.write(content)
    print(f"[ok] Patched {TARGET}")
    print("     Restart the streamer to pick up the change.")


if __name__ == "__main__":
    main()
