"""Train a YOLO pill detector and export it as the Core ML model the app
auto-discovers. Run this on a machine with a GPU and normal internet access
(it does not run in restricted/offline environments).

Setup
-----
    pip install ultralytics

Dataset
-------
YOLO-format dataset (images + one .txt per image with `class cx cy w h`
normalized rows) described by a data.yaml:

    path: /data/pills
    train: images/train
    val: images/val
    names:
      0: pill

Sources, best first:
  1. Photos of YOUR pharmacy's stock on YOUR trays (a few hundred photos,
     20-60 pills each, varied lighting/angles), labeled with Roboflow or
     Label Studio. This is what makes the model accurate on the pills you
     actually dispense.
  2. Public pill datasets (search Roboflow Universe for "pill detection";
     several thousand labeled images exist). Check each dataset's license.
  3. Both: public dataset for pretraining breadth + your photos for the
     final fine-tune.

Label every pill as one class ("pill"). Include hard negatives: empty
trays, trays with glare, hands, bottles.

Usage
-----
    python3 tools/train_pill_yolo.py /data/pills/data.yaml

Output: PillDetector.mlpackage — drag it into Xcode, tick the PillCount
target, rebuild. The app detects the model at launch and switches to
hybrid mode (ML primary + classical cross-check) automatically.
"""
import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    data_yaml = sys.argv[1]

    from ultralytics import YOLO

    # Nano model: ~6 MB in the app bundle, 30+ fps on the Neural Engine.
    # Bump to yolo11s if calibration-mode accuracy plateaus.
    model = YOLO("yolo11n.pt")

    model.train(
        data=data_yaml,
        epochs=120,
        imgsz=640,
        batch=16,
        patience=25,
        # Counting cares about every instance; keep augmentation realistic
        # for a top-down tray (no vertical-flip asymmetry concerns).
        degrees=180, flipud=0.5, fliplr=0.5,
        scale=0.4, translate=0.1,
        mosaic=1.0, close_mosaic=15,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.35,
    )

    metrics = model.val(data=data_yaml)
    print(f"validation mAP50: {metrics.box.map50:.3f}  "
          f"mAP50-95: {metrics.box.map:.3f}")

    # nms=True bakes non-max suppression into the Core ML pipeline so iOS
    # Vision returns ready-to-use VNRecognizedObjectObservation boxes.
    exported = model.export(format="coreml", nms=True, imgsz=640)

    out = Path("PillDetector.mlpackage")
    if out.exists():
        shutil.rmtree(out)
    shutil.move(exported, out)
    print(f"\nWrote {out.resolve()}")
    print("Drag it into Xcode (tick the PillCount target) and rebuild — "
          "the app switches to hybrid ML+CV mode automatically.")
    print("Then re-run the in-app calibration protocol before relying on it.")


if __name__ == "__main__":
    main()
