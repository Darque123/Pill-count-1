# Training brief: produce PillDetector.mlpackage for the PillCount app

**Audience:** a Claude Code session (or a human) on a machine with a GPU and
unrestricted internet. Everything needed is in this repo; this file is the
complete work order. When you finish, the app in this repo switches to
hybrid ML+CV mode automatically.

## Mission

Train a single-class ("pill") YOLO object detector on **openly licensed**
pill photos, validate it against this repo's test assets, export it to
Core ML, and install it into the Xcode project. The model must:

1. Detect individual pills (round tablets, oblong caplets, capsules) on
   pharmacy-tray-style backgrounds, viewed from above.
2. Run on an iPhone Neural Engine in real time (use the nano model size;
   the app processes ~15 fps and drops frames while busy, so ≤60 ms per
   frame on an A15 is comfortable).
3. Ship with a completed `training/MODEL_CARD.md` documenting datasets,
   licenses, metrics, and export settings — this is a pharmacy tool;
   provenance is not optional.

## Context you need (read these first)

- `README.md` → "Optional ML detector (hybrid mode)": how the app consumes
  the model. Short version: the app looks for a compiled Core ML model
  named `PillDetector` in its bundle; when present, ML becomes the primary
  detector and the classical OpenCV pipeline becomes a live cross-check
  with an on-screen disagreement warning.
- `PillCount/Detection/MLPillDetector.swift`: the consumer. It expects a
  YOLO detector exported with **NMS baked in** (`nms=True`) so Vision
  returns `VNRecognizedObjectObservation` boxes directly, and it uses
  `.scaleFill` — export at `imgsz=640`.
- `docs/LIMITATIONS.md`: what the model is and isn't expected to fix.
  Stacked pills stay unsolvable; do not chase them with training data.

## Step 1 — Source openly licensed data

Rules:
- Every dataset you use must have an explicit open license (CC BY 4.0,
  CC0, Apache-2.0, MIT, US-government public domain). **No license file =
  do not use.** Record each source + license in the model card.
- Prefer multi-pill, top-down, tray-like scenes. Single-pill catalog
  closeups are only useful as compositing material (see synthesis below).

Places to look (verify licenses at fetch time — listings change):
1. **Roboflow Universe** (universe.roboflow.com): search "pill detection",
   "pill counting", "tablet detection". Many datasets are CC BY 4.0 with
   thousands of labeled multi-pill images, downloadable in YOLO format
   directly (free account/API key).
2. **Kaggle datasets** (kaggle.com/datasets): search "pill detection",
   "pill counting". Check per-dataset license.
3. **NIH/NLM C3PI reference images** (US government work, public domain):
   single-pill studio photos of thousands of US medications. Not directly
   usable as detection labels, but ideal cutout material for **synthetic
   scene generation**: composite N pill cutouts onto tray-colored
   backgrounds at random poses (touching allowed, overlap ≤ 15%), with
   contact shadows, glare blobs, and illumination gradients — the label
   boxes come free. `tools/test_synthetic.py` shows the scene styles that
   matter; a compositor that mirrors those hard cases with *real* pill
   textures is the highest-leverage data you can add.

Target: **≥ 3,000 labeled multi-pill images** total (real + synthetic),
with hard negatives included (empty trays, hands, bottles, glare). Merge
everything into one YOLO dataset with a single class `pill` (remap any
per-drug classes to class 0), split ~90/10 train/val **by scene, not by
frame**, and write `data.yaml`.

## Step 2 — Train and export

```bash
pip install ultralytics
python3 tools/train_pill_yolo.py /path/to/data.yaml
```

The script trains `yolo11n`, prints validation mAP, and exports
`PillDetector.mlpackage` (Core ML, `nms=True`, `imgsz=640`). Read its
docstring for the tuned augmentation rationale before changing anything.

## Step 3 — Acceptance gates (all must pass)

```bash
# 1. Detector metrics on the val split: mAP50 >= 0.90.
#    (train_pill_yolo.py prints this.)

# 2. Counting accuracy on this repo's fixtures — the number that actually
#    matters for dispensing is EXACT COUNT, not mAP:
python3 training/eval_model.py runs/detect/train/weights/best.pt tools/fixtures
#    Gate: >= 6/8 exact. (The synthetic look differs from photos; if the
#    model was trained with a synthetic slice, expect 8/8.)

# 3. Real footage — download the public test clip and check the smoother
#    locks only at the true count (12):
#    clip: old/pill1.avi in https://github.com/kien-ly/count-drug
python3 training/eval_model.py runs/detect/train/weights/best.pt pill1.avi --truth 12
#    Gate: locks at 12, never locks at any other value.

# 4. Size and latency: the .mlpackage should be <= 15 MB (nano is ~6 MB).
```

If a gate fails, iterate on data (more real multi-pill scenes, better
synthetic hard cases) before touching hyperparameters.

## Step 4 — Install the model into the app

Preferred (Xcode): drag `PillDetector.mlpackage` into the Xcode project
navigator under `PillCount/Detection/`, tick the *PillCount* target,
build. Xcode compiles it to `PillDetector.mlmodelc` in the bundle, which
`MLPillDetector` discovers at launch.

Headless (editing `PillCount.xcodeproj/project.pbxproj` directly — the
file uses hand-assigned `D1CE...` UUIDs; continue the sequence):

1. Copy the model to `PillCount/Detection/PillDetector.mlpackage`.
2. PBXFileReference section:
   `D1CE00000000000000000032 /* PillDetector.mlpackage */ = {isa = PBXFileReference; lastKnownFileType = folder.mlpackage; path = PillDetector.mlpackage; sourceTree = "<group>"; };`
3. PBXBuildFile section (Core ML models compile via the **Sources** phase):
   `D1CE0000000000000000004F /* PillDetector.mlpackage in Sources */ = {isa = PBXBuildFile; fileRef = D1CE00000000000000000032 /* PillDetector.mlpackage */; };`
4. Add `D1CE00000000000000000032` to the `Detection` group's children and
   `D1CE0000000000000000004F` to the `Sources` build phase file list.

Commit the .mlpackage (it's small) together with the completed
`training/MODEL_CARD.md` (template in this directory) and the eval outputs.

## Step 5 — Hand back

Final checklist for your closing summary:
- [ ] Datasets + licenses enumerated in MODEL_CARD.md
- [ ] mAP50, fixtures exact-count, and video-lock results reported
- [ ] .mlpackage committed and wired into the Xcode project
- [ ] Reminder to the user: re-run the **in-app calibration protocol**
      (docs/LIMITATIONS.md) on their real pills before relying on the
      hybrid mode — training metrics are not a dispensing validation.
