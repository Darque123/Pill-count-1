# Known limitations & accuracy assumptions

An unvalidated count is a patient-safety risk. This page states what the
detector assumes, where it is known to fail, and how to measure its accuracy
before anyone relies on it.

## Operating assumptions

The pipeline was designed and tested for the standard pharmacy workflow:

- Pills lie **in a single layer** on a counting tray, viewed roughly from
  above (phone approximately parallel to the tray).
- The tray is a **plain, roughly uniform surface**. Both light and dark trays
  work; a matte tray in a contrasting shade to the pills is ideal.
- Pills in one count are visible and not buried under each other.
- Reasonable indoor lighting. Gradients and soft shadows are removed by
  illumination flattening; the torch button helps in dim light.

## Known failure modes

| Case | Behavior | Mitigation |
|---|---|---|
| **Stacked / heavily overlapping pills** (one pill partly on top of another) | The hidden pill undercounts — a silhouette-based method cannot see it | Spread pills to a single layer (standard tray practice); verify outlines on the frozen frame |
| **Side-by-side identical caplets in full contact** | Separated when the contact seam is visible (the usual case — convex pills cast a contact shadow); may merge as one if the seam is optically invisible | Outlines make a merge obvious; nudge with the spatula or use +/− override |
| **Deeply scored tablets** under harsh light | A strongly shadowed score line can trigger the seam-splitting logic and count one tablet as two | Outlines show a split pill clearly; set `valleyAssist = false` in `PillPipeline.hpp` if calibration shows this dominating your error |
| **Pill-sized glare blobs** on glossy trays | A specular highlight the size and shape of a pill can be counted as one | Use a matte tray, tilt slightly, or use the torch to even out lighting; empty-tray checks in calibration mode will catch a tray that glares |
| **Pills at the frame edge** | A pill partially out of frame may be filtered out by shape checks or counted with a clipped outline | Keep the whole tray in view with margin |
| **Transparent / translucent capsules** | Weak contrast in all three cues (luminance, saturation, rim) can drop them | Dark tray recommended; validate in calibration mode |
| **Very small pills at high camera distance** | Below the minimum-area filter at the 640 px working resolution | Move closer, or raise `maxDimension` / lower `minAreaFraction` (costs frames per second) |
| **Pills filling most of the frame** (extreme close-up / dense macro shot) | The background estimate is dragged toward the pills, eroding their masks — undercounts, ragged outlines | Frame the whole tray, not a pill pile; the camera's minimum focus distance makes this hard to hit accidentally in the app |
| **Motion blur** | Counts during motion are unreliable — by design the count won't lock while values disagree | Wait for LOCKED ✓; it requires 8 consecutive agreeing frames on a steady scene |

## Accuracy claims — what has and hasn't been verified

**Verified (automated, reproducible):** the shipping C++ pipeline counts
exactly on 8 synthetic hard-case scenes (touching 7-pill cluster,
side-by-side caplets, white-on-light with illumination gradient, glare +
shadow, mixed sizes touching, dark tray, empty tray) and on 160 randomized
variations of them — `tools/` re-runs this in one command.

**Verified on real footage:** a 10-second handheld video of 12 white scored
caplets on a dark table (several touching, visible score lines — see
`old/pill1.avi` in [kien-ly/count-drug](https://github.com/kien-ly/count-drug)):
the smoother locks at 12 — the correct count — within 0.25 s, and **never
locks at any other value** for the whole clip (momentary undercounts during
camera shake stay in "counting…" and are never committed). Reproduce with
`python3 tools/eval_video.py pill1.avi 12`. This footage drove several real
fixes the synthetic suite missed (all covered by tests now): the rim-
gradient fill gluing touching clusters, the flattening halo joining
opposite-polarity masks, and score-line over-splitting — which is why the
valley threshold is proportional to local brightness (contact shadows
measure ~35–50 % of pill brightness, score grooves ~15–25 %), suppressed
near specular glare, and why mask components with soft (penumbra-like)
boundaries are rejected: physical pills always have crisp silhouettes.

**Still not verified:** breadth. One real pill type, one tray, one lighting
setup is a smoke test, not a validation. Real deployment still requires the
calibration-mode protocol below across your pill types, trays, and lighting.

**Before real-world reliance, run a validation protocol** with the built-in
calibration mode (checklist button):

1. For each pill type you dispense: place a hand-verified known count on the
   tray (vary count sizes, e.g. 10 / 30 / 60 / 90).
2. Freeze, enter the true count, repeat across lighting conditions and tray
   positions. Include empty-tray frames.
3. Review the stats panel: **exact-count rate** is the number that matters
   for dispensing (mean error hides ±1 misses). Export the CSV for records.
4. Re-run after any parameter change or app update.

Overrides are also logged: a pill type that repeatedly needs manual +/− is a
pill type the detector is weak on — calibrate it specifically.

## Design guardrails already in place

- The count **locks only when stable** across 8 consecutive frames; the UI
  is explicit about when the number is not yet trustworthy.
- **Every counted pill is outlined** so a merge (one outline over two pills)
  or a phantom (outline over nothing) is visually obvious on the frozen frame.
- The marker stage **cannot drop a detected mask component** (each is
  guaranteed a watershed seed); undercounts come from masking/overlap, not
  from marker bookkeeping.
- An **empty tray counts 0**: every cue self-disables below a noise floor
  rather than letting Otsu promote sensor noise to "pills".
- All processing is **on-device**; no image or count ever leaves the phone
  unless the user explicitly shares the CSV log.

## If classical CV plateaus

Heavy overlap (pills on pills) is the main case where this silhouette-based
approach hits a ceiling. The clean extension point is `DetectionEngine`: a
lightweight class-agnostic instance segmenter (e.g. a small YOLO variant or
a distilled Segment-Anything model compiled to Core ML, running fully
offline) could replace `PillPipeline` as the region proposer for exactly
those frames — keeping the classical path as the default and the same
smoothing/UX safety layer on top. Costs to weigh: model size in the app
bundle, per-frame latency on older devices, and losing the "no training,
any pill" guarantee unless the model is genuinely class-agnostic.
