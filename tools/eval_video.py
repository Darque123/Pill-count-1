"""Run the detection pipeline + the app's temporal smoother over a video of
pills, exactly as the iOS app would process a live camera feed.

Usage:
    python3 tools/eval_video.py path/to/video.mp4 [true_count]

Prints the per-frame count histogram and the smoothed lock timeline; writes
an annotated copy next to the input (<name>_annotated.gif, downsampled).
If true_count is given, exits non-zero unless every locked value equals it.

Example real-world footage (white scored caplets on a dark table, ~10 s
handheld): old/pill1.avi in https://github.com/kien-ly/count-drug —
the pipeline locks at the correct count of 12 within 0.25 s and holds it
for the entire clip.
"""
import os
import sys
from collections import Counter

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import detect, Params  # noqa: E402

GREEN, YELLOW = (80, 235, 60), (60, 220, 250)


class Smoother:
    """Python replica of the app's CountSmoother (lockFrames=8, breakFrames=3)."""

    def __init__(self, lock=8, brk=3):
        self.lock, self.brk = lock, brk
        self.run_v, self.run_n, self.locked = -1, 0, None

    def ingest(self, count):
        if count == self.run_v:
            self.run_n += 1
        else:
            self.run_v, self.run_n = count, 1
        if self.run_n >= self.lock:
            self.locked = self.run_v
        elif (self.locked is not None and self.run_v != self.locked
              and self.run_n >= self.brk):
            self.locked = None
        if self.locked is not None:
            return "locked", self.locked
        return "counting", self.run_v


def annotate(frame, res, color, label, badge, badge_color):
    scale = frame.shape[1] / res.work_size[0]
    out = frame.copy()
    for d in res.detections:
        contour = (d.contour * scale).astype("int32").reshape(-1, 1, 2)
        cv2.polylines(out, [contour], True, color, 2, cv2.LINE_AA)
    cv2.putText(out, label, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(out, label, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(out, (12, 56), (28 + tw, 70 + th), badge_color, -1)
    cv2.putText(out, badge, (20, 62 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    path = sys.argv[1]
    truth = int(sys.argv[2]) if len(sys.argv) > 2 else None

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    params, smoother = Params(), Smoother()
    counts, locked_values, annotated = [], Counter(), []
    first_lock = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = detect(frame, params)
        state, value = smoother.ingest(res.count)
        counts.append(res.count)
        if state == "locked":
            locked_values[value] += 1
            if first_lock is None:
                first_lock = len(counts) - 1
        annotated.append(annotate(
            frame, res,
            GREEN if state == "locked" else YELLOW,
            str(value),
            "LOCKED" if state == "locked" else "counting...",
            (60, 180, 50) if state == "locked" else (0, 149, 255)))
    cap.release()

    n = len(counts)
    print(f"{n} frames @ {fps:.0f} fps")
    print("raw per-frame counts:", dict(sorted(Counter(counts).items())))
    print(f"locked on {sum(locked_values.values())}/{n} frames, "
          f"values: {dict(locked_values)}")
    if first_lock is not None:
        print(f"first lock: frame {first_lock} ({first_lock / fps:.2f}s)")

    try:
        from PIL import Image
        small = [Image.fromarray(cv2.cvtColor(
                     cv2.resize(f, None, fx=0.75, fy=0.75), cv2.COLOR_BGR2RGB))
                 for f in annotated[::3]]
        out = os.path.splitext(path)[0] + "_annotated.gif"
        small[0].save(out, save_all=True, append_images=small[1:],
                      duration=int(3000 / fps), loop=0, optimize=True)
        print("wrote", out)
    except ImportError:
        print("(pip install pillow for an annotated gif)")

    if truth is not None:
        bad = [v for v in locked_values if v != truth]
        if bad or not locked_values:
            print(f"FAIL: locked values {dict(locked_values)} != {truth}")
            sys.exit(1)
        print(f"PASS: every locked value == {truth}")


if __name__ == "__main__":
    main()
