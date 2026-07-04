# Model card: PillDetector

> Copy this file to `training/MODEL_CARD.md`, fill in every field, and
> commit it alongside the .mlpackage. A pharmacy tool must be able to
> answer "what is this model and where did its data come from" — an
> unfilled field means the model does not ship.

## Summary
- **Architecture:** (e.g. YOLO11n, single class "pill")
- **Trained:** (date, hardware, wall-clock)
- **Exported:** Core ML, `nms=True`, `imgsz=640` (from tools/train_pill_yolo.py)
- **File:** PillDetector.mlpackage, ___ MB

## Training data
| Source | Images | License | Notes |
|---|---|---|---|
| (dataset name + URL) | | (must be an explicit open license) | |
| (synthetic compositor, if used) | | n/a (generated) | describe pill-cutout source + its license |

Hard negatives included: (empty trays / hands / glare / ...)
Train/val split method: (by scene, ratio)

## Metrics
- Val mAP50: ___  mAP50-95: ___
- Repo fixtures (`training/eval_model.py best.pt tools/fixtures`): ___/8 exact, MAE ___
- Real clip (pill1.avi, truth 12): locks only at ___ ✅/❌
- Latency on device (model compute time, if measured): ___ ms on ___

## Known weaknesses
- (pill types / scenes where it underperforms, from eval + calibration)

## Safety notes
- Stacked/hidden pills are undetectable by design; workflow rule applies.
- The classical OpenCV cross-check remains active in the app; disagreement
  shows an on-screen "verify" warning.
- In-app calibration protocol re-run after install: ✅/❌ (date, results)
