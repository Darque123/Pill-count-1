"""Acceptance harness for a trained pill detector: measures COUNTING
accuracy (the metric that matters for dispensing) on this repo's test
assets, and replays the app's temporal smoother on video.

Usage:
    # images + manifest.csv ("file,truth" rows, as in tools/fixtures):
    python3 training/eval_model.py best.pt tools/fixtures [--conf 0.35]

    # video with a known true count (app smoother must lock ONLY there):
    python3 training/eval_model.py best.pt pill1.avi --truth 12

Accepts any model ultralytics can load (.pt, .onnx, exported CoreML on
macOS). Exit code 0 = gates passed.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from eval_video import Smoother  # noqa: E402  (app CountSmoother replica)


def count(model, source, conf):
    results = model.predict(source, conf=conf, verbose=False)
    return len(results[0].boxes)


def eval_images(model, dataset_dir, conf):
    manifest = os.path.join(dataset_dir, "manifest.csv")
    if not os.path.exists(manifest):
        print(f"no manifest.csv in {dataset_dir}")
        return 1
    total = exact = 0
    abs_err = 0
    for line in open(manifest):
        line = line.strip()
        if not line:
            continue
        name, truth = line.rsplit(",", 1)
        truth = int(truth)
        got = count(model, os.path.join(dataset_dir, name), conf)
        ok = got == truth
        exact += ok
        total += 1
        abs_err += abs(got - truth)
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} truth={truth:3d} got={got:3d}")
    print(f"{exact}/{total} exact, MAE {abs_err / max(total, 1):.2f}")
    return 0 if exact >= total * 0.75 else 1


def eval_video_file(model, path, truth, conf):
    import cv2
    cap = cv2.VideoCapture(path)
    smoother = Smoother()
    locked_values = Counter()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, conf=conf, verbose=False)
        state, value = smoother.ingest(len(results[0].boxes))
        if state == "locked":
            locked_values[value] += 1
        n += 1
    cap.release()
    print(f"{n} frames; locked values: {dict(locked_values)}")
    if truth is None:
        return 0
    bad = [v for v in locked_values if v != truth]
    if bad or not locked_values:
        print(f"FAIL: expected locks only at {truth}")
        return 1
    print(f"PASS: locks only at {truth}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to .pt / .onnx model")
    ap.add_argument("source", help="fixtures dir (with manifest.csv) or video")
    ap.add_argument("--truth", type=int, default=None,
                    help="true count for a video source")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    if os.path.isdir(args.source):
        sys.exit(eval_images(model, args.source, args.conf))
    sys.exit(eval_video_file(model, args.source, args.truth, args.conf))


if __name__ == "__main__":
    main()
