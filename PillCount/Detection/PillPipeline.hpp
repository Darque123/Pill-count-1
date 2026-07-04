//
//  PillPipeline.hpp
//  PillCount
//
//  Pure C++ pill-detection pipeline. This file has no Apple dependencies so
//  the exact shipping algorithm can also be compiled and regression-tested
//  off-device (see tools/cpp_test). All tunable parameters live in Params.
//
//  A Python reference implementation with identical steps lives in
//  tools/pipeline.py — keep the two in sync when tuning.
//

#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace pillcount {

/// Tunable detection parameters. Defaults were validated against the
/// synthetic hard-case suite in tools/test_synthetic.py (touching clusters,
/// white-on-white, glare, shadows, mixed sizes).
struct Params {
    /// Frames are resized so max(width, height) == maxDimension before
    /// processing (downscaled for speed; small inputs upscaled, capped at
    /// 5x, so kernel sizes stay proportional to pill sizes).
    int maxDimension = 640;

    /// Illumination flattening: background estimated with a Gaussian blur of
    /// kernel size (bgSmoothFraction * maxDimension). Removes lighting
    /// gradients, soft shadows, and smooth reflected-light glow so the tray
    /// maps to a uniform gray.
    double bgSmoothFraction = 0.25;

    /// Pre-threshold Gaussian blur kernel (odd).
    int blurKernel = 5;

    /// If Otsu on the |deviation-from-tray| image picks a threshold below
    /// this, the frame is treated as having no luminance foreground —
    /// prevents phantom detections on an empty tray.
    int minDeviation = 8;

    /// The saturation mask is only trusted when Otsu on the saturation
    /// channel is at least this (i.e. there is genuine color in the scene).
    int minSaturation = 40;

    /// The rim-shading (Sobel edge) cue is only trusted when Otsu on the
    /// gradient magnitude is at least this.
    int minGradient = 18;
    /// Kernel used to close edge fragments into fillable rings.
    int gradCloseKernel = 7;

    /// Accepted pill area, as a fraction of the working-frame pixel count.
    double minAreaFraction = 0.0005;
    double maxAreaFraction = 0.05;

    /// Shape filters on final contours: pills are convex-ish and not
    /// extremely elongated.
    double minSolidity = 0.70;
    double maxAspect = 4.5;

    /// Watershed markers: a pixel seeds "sure foreground" when its distance-
    /// transform value >= distRatio * (local max within markerWindow). The
    /// local-normalized threshold separates touching pills of mixed sizes.
    double distRatio = 0.45;
    double markerWindowFraction = 0.09;   // window = fraction * maxDimension

    /// Valley assist: subtract thin dark contact seams (black-hat) from the
    /// marker mask so touching pills seed separate markers. The depth
    /// required scales with local brightness: contact shadows run ~35-50%
    /// of pill brightness, score-line grooves ~15-25% (measured on real
    /// footage of scored white caplets), so contacts are cut without
    /// splitting scored tablets. Disable if calibration mode still shows
    /// score-line over-splitting on your stock.
    bool valleyAssist = true;
    double valleyRatio = 0.35;   // required black-hat depth / local level
    int valleyFloor = 15;        // absolute minimum depth (noise gate)
    int valleyKernel = 9;        // must span shadow-junction zones in tight
                                 // clusters without deepening the response
                                 // to score-line grooves

    /// Minimum median gray-image gradient along a mask component's boundary.
    /// Physical objects have crisp silhouettes (pills measure 40-430);
    /// penumbra fragments of cast shadows measure 5-15.
    int minBoundaryGrad = 25;

    /// Cleanup morphology kernel sizes.
    int openKernel = 3;
    int closeKernel = 3;
};

/// One detected pill.
struct Pill {
    /// Contour polygon in normalized [0,1] coordinates of the frame.
    std::vector<cv::Point2f> contour;
    /// Centroid, normalized [0,1].
    cv::Point2f center;
    /// Contour area as a fraction of the frame area.
    double areaFraction = 0.0;
};

struct FrameResult {
    std::vector<Pill> pills;
    /// Size of the downscaled working frame (for diagnostics).
    cv::Size workSize;
};

/// Detect and count pills in a BGR frame. Thread-safe (no shared state).
FrameResult detectPills(const cv::Mat& bgr, const Params& params);

}  // namespace pillcount
