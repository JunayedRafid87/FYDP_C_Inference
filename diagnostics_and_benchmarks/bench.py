#!/usr/bin/env python3
"""
bench.py — time PURE inference on every ONNX model in a directory.

No camera, no preprocessing, no JPEG encoding, no web server. Synthetic
input straight into the session. The point is to separate "the model is
slow" from "everything around the model is slow" — those have completely
different fixes, and the live pipeline can't tell you which you have.

Also sweeps ONNX Runtime thread counts. The default is often well below
the core count, and on an 8-core board that alone can cost 2x.

Usage:
    python3 bench.py                 # every *.onnx in the current dir
    python3 bench.py /path/to/dir
    python3 bench.py model.onnx      # one model, thread sweep

Needs: pip3 install onnxruntime    (no ultralytics, no torch)
"""

import glob
import os
import sys
import time

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime missing — pip3 install onnxruntime")

RUNS = 20
WARMUP = 3


def bench(path, threads=None):
    """Return (ms_median, ms_best, input_shape) for one model."""
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    blob = np.random.rand(*shape).astype(np.float32)

    for _ in range(WARMUP):
        sess.run(None, {inp.name: blob})

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: blob})
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2], times[0], shape


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."

    if os.path.isdir(target):
        models = sorted(glob.glob(os.path.join(target, "*.onnx")))
    else:
        models = [target]

    if not models:
        sys.exit(f"no .onnx files found in {target}")

    cores = os.cpu_count()
    print(f"onnxruntime {ort.__version__} | {cores} cores | "
          f"{RUNS} runs after {WARMUP} warmup\n")

    print(f"{'model':44} {'input':18} {'median':>9} {'best':>9} {'fps':>7}")
    print("-" * 92)

    results = []
    for m in models:
        try:
            med, best, shape = bench(m)
        except Exception as e:
            print(f"{os.path.basename(m):44} FAILED: {e}")
            continue
        name = os.path.basename(m)
        results.append((name, med))
        print(f"{name:44} {str(shape):18} {med:7.1f}ms {best:7.1f}ms "
              f"{1000/med:6.1f}")

    # Thread sweep on the fastest model — usually the smallest input size,
    # which is the one you would actually deploy.
    if results:
        fastest = min(results, key=lambda r: r[1])[0]
        path = fastest if os.path.isfile(fastest) else os.path.join(target, fastest)
        print(f"\nthread sweep on {fastest}")
        print("-" * 92)
        for t in (1, 2, 4, cores):
            try:
                med, best, _ = bench(path, threads=t)
                print(f"  intra_op_num_threads={t:<3} {med:7.1f}ms  "
                      f"({1000/med:.1f} fps)")
            except Exception as e:
                print(f"  threads={t}: {e}")

    print("\nInterpretation:")
    print("  If median inference is well under your live frame time, the")
    print("  model is not the bottleneck — preprocessing (CLAHE, bicubic")
    print("  upscale, colormap), JPEG encoding, or frame buffering is.")
    print("  Two models running concurrently will each be slower than the")
    print("  figures above, since they compete for the same cores.")


if __name__ == "__main__":
    main()
