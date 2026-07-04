# PillCount

Real-time pill counting for pharmacy trays, on iPhone. Point the camera at
pills on a counting tray and the app detects, outlines, and counts every pill
live — for any pill type, with **no per-pill training or configuration** and
**no network access** (all processing is on-device OpenCV).

| | |
|---|---|
| Platform | iOS 16+, Swift + SwiftUI, AVFoundation |
| Vision | Classical CV via OpenCV (C++), no ML model, fully offline |
| Detection | Illumination flattening → multi-cue segmentation → watershed separation of touching pills → contour filtering |
| Anti-flicker | Count locks only after 8 consecutive agreeing frames — explicit **LOCKED ✓** vs **counting…** indicator |
| Safety | Per-pill outlines for visual verification, freeze-frame, logged manual override, built-in accuracy/calibration mode |

## Building

Requirements: Xcode 15 or newer, an iPhone running iOS 16+ (the camera
pipeline needs real hardware; the Simulator has no camera).

1. Open `PillCount.xcodeproj` in Xcode.
2. Let Xcode resolve Swift Package dependencies. OpenCV is integrated via
   **Swift Package Manager** using [`yeatse/opencv-spm`](https://github.com/yeatse/opencv-spm)
   (pinned `4.10.0 ..< 5.0.0`), which downloads the prebuilt official
   `opencv2.xcframework` — no manual OpenCV build, no CocoaPods.
3. Select the *PillCount* target → *Signing & Capabilities* → choose your
   development team (the bundle id `com.example.pillcount` is a placeholder —
   change it to something unique for your team).
4. Build and run on a device. Grant camera permission when asked.

**If SPM can't download the binary** (offline/firewalled build machine):
download `opencv-4.x-ios-framework.zip` from the
[official OpenCV releases](https://github.com/opencv/opencv/releases), unzip,
remove the `opencv2` package dependency from the project, and drag
`opencv2.xcframework` into the target's *Frameworks, Libraries, and Embedded
Content* (Do Not Embed — it's a static framework). No code changes are
needed; the source includes only standard `<opencv2/…>` headers.

### How OpenCV is wired in

```
Swift UI / camera code
   │  (bridging header: PillCount/Detection/PillCount-Bridging-Header.h)
   ▼
OpenCVWrapper.h / .mm       Objective-C++ adapter: CVPixelBuffer ⇄ cv::Mat,
   │                        boxes results for Swift
   ▼
PillPipeline.hpp / .cpp     Pure C++ detection pipeline (no Apple deps) —
                            the same file compiles on Linux/macOS for the
                            regression tests in tools/
```

## Using the app

- **Live counting** — hold the phone roughly flat above the tray so pills
  don't occlude each other. Every detected pill is outlined; the big number
  is the count. **Trust the count only when the badge shows green
  "LOCKED ✓"** — orange "counting…" means the number hasn't been stable
  across consecutive frames yet. Tap the preview to focus/expose on the tray;
  the flashlight button can rescue dim or glare-heavy counters.
- **Freeze** (pause button) — locks the current frame and count so you can
  verify outlines pill-by-pill against the still image.
- **Manual override** (+/− while frozen) — corrects an edge case. Every
  override is logged with the machine count so systematic detector errors are
  visible in the accuracy record.
- **Multi-tray totals** — for counts larger than one tray: freeze and verify
  a tray, tap **"Add N to total"**, pour the pills into the bottle, lay out
  the next tray, and repeat. The bottle total and tray count stay on screen,
  with undo-last-tray and reset (confirmation required). Every committed
  tray is written to the audit log with its machine count and any manual
  adjustment. Requiring a frozen, inspectable frame before a tray can enter
  the total is deliberate — nothing joins the total unverified.
- **Accuracy / calibration mode** (checklist button) — freeze a frame, count
  the pills by hand, enter the true count; the app records its own error.
  The stats panel aggregates checks (exact-count rate, mean absolute error,
  bias) and the log exports as CSV. **Run this against known counts across
  your pill types before relying on the app** — see
  [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## How detection works

For each frame (downscaled to ≤640 px for real-time processing; results are
drawn over the full-resolution preview):

1. **Illumination flattening** — grayscale is divided by a heavily blurred
   copy of itself, mapping the tray to a uniform gray regardless of lighting
   gradients and soft shadows.
2. **Multi-cue foreground mask** — a pixel is foreground if it deviates from
   the tray level in *luminance* (Otsu on |flattened − tray|, catching pills
   both lighter and darker than the tray in one frame), *saturation*
   (colored pills whose brightness matches the tray), or *rim gradient*
   (white-on-white pills still show shaded rims because they're convex).
   Each cue self-disables when its Otsu threshold falls below a noise floor,
   so an empty tray counts 0 instead of hallucinating pills from noise.
3. **Junk rejection** — mask components thinner than the minimum pill radius
   (shadow edges, tray ridges, glare streaks) are erased.
4. **Touching-pill separation** — watershed markers come from the distance
   transform thresholded against its *local* neighborhood maximum (so mixed
   pill sizes work), computed on a mask with dark contact seams subtracted
   (so side-by-side pills seed separate markers). Every component is
   guaranteed at least one marker, so the marker stage can never drop a pill.
   Watershed then floods on the color frame, snapping boundaries to the
   shading seam between touching pills.
5. **Contour filtering** — per-region contours filtered by area, solidity,
   and aspect ratio.
6. **Temporal smoothing** — the raw per-frame count feeds `CountSmoother`:
   the UI locks only after 8 consecutive identical counts (~0.5–1 s) and
   drops a lock only after 3 consecutive disagreeing frames, so the number
   never flickers frame-to-frame.

All tunables (area thresholds, smoothing window, marker ratios…) are in
`PillPipeline.hpp` (`pillcount::Params`) with per-parameter docs, mirrored to
Swift via `PCDetectorParams`.

## Testing the pipeline without a device

The exact C++ that ships in the app is regression-tested against synthetic
tray scenes with known ground truth — including the hard cases: a 7-pill
touching cluster, side-by-side touching caplets, white pills on a light tray
under an illumination gradient, glare + hard shadows, mixed sizes, and an
empty tray.

```bash
# Python reference implementation (same algorithm, annotated):
pip install opencv-python-headless numpy
python3 tools/test_synthetic.py            # 8 scenes, asserts exact counts

# The shipping C++ against the same scenes (needs desktop OpenCV, e.g.
# `apt install libopencv-dev` / `brew install opencv`):
python3 tools/test_synthetic.py --dump tools/fixtures
g++ -std=c++17 -O2 tools/cpp_test/main.cpp PillCount/Detection/PillPipeline.cpp \
    -IPillCount/Detection $(pkg-config --cflags --libs opencv4) -o /tmp/pill_test
/tmp/pill_test tools/fixtures
```

Current status: **8/8 scenes exact, 155/160 across 20 randomized scene
variations** (the only miss: a hexagonally packed 7-pill cluster at −1 in
5 variations, always visible as a merged outline), at ~45 ms/frame on a
desktop CPU at 640 px working resolution. On real handheld footage of 12
white scored caplets on a dark table (several touching), the smoother
locks at 12 within 0.25 s and **never locks at a wrong value** — reproduce
with `tools/eval_video.py` (details and source of the footage in
[docs/LIMITATIONS.md](docs/LIMITATIONS.md)).

Synthetic scenes and one video are not a substitute for broad real-world
validation — use the in-app calibration mode with your pills, trays, and
lighting.

## Repository layout

```
PillCount/                  the iOS app
  App/                      entry point, root view, app state machine
  Camera/                   AVCaptureSession + SwiftUI preview
  Detection/                ← counting logic, isolated & tunable
    PillPipeline.hpp/.cpp   pure C++ OpenCV pipeline (all parameters here)
    OpenCVWrapper.h/.mm     ObjC++ bridge to Swift
    DetectionEngine.swift   frame pump (drops frames while busy)
    CountSmoother.swift     temporal anti-flicker / lock state
  Calibration/              accuracy log + calibration UI
  UI/                       overlay & HUD views
tools/                      Python reference pipeline + synthetic test suite
tools/cpp_test/             Linux/macOS harness for the shipping C++
docs/LIMITATIONS.md         known limitations & accuracy assumptions — read this
```

## Safety notes

This is a counting aid, not a replacement for pharmacist verification. The
per-pill outlines, the LOCKED indicator, freeze-frame, logged overrides, and
calibration mode exist so a human can and should verify every dispense.
Review [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before any real-world use.
