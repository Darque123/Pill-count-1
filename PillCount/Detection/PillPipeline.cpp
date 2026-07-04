//
//  PillPipeline.cpp
//  PillCount
//
//  Classical-CV pill detection: illumination flattening -> multi-cue
//  foreground mask (luminance deviation + saturation + rim gradient) ->
//  morphology -> adaptive distance-transform markers -> watershed ->
//  contour filtering. See PillPipeline.hpp for the parameter reference and
//  tools/pipeline.py for the annotated Python twin.
//

#include "PillPipeline.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace pillcount {
namespace {

int makeOdd(double v) {
    int n = std::max(1, static_cast<int>(v));
    return (n % 2 == 1) ? n : n + 1;
}

cv::Mat ellipseKernel(int size) {
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
}

/// Divide grayscale by a heavily blurred copy: the tray maps to ~128
/// regardless of lighting gradients and soft shadows.
cv::Mat flattenIllumination(const cv::Mat& gray, const Params& p) {
    int k = makeOdd(std::max(gray.rows, gray.cols) * p.bgBlurFraction);
    cv::Mat bg, flat;
    cv::GaussianBlur(gray, bg, cv::Size(k, k), 0);
    cv::divide(gray, bg, flat, 128.0);
    return flat;
}

/// Remove non-pill-like components. Pills (and clusters of touching pills)
/// are solid, reasonably thick blobs; this rejects:
///   - thin structures (max inscribed radius < min pill radius): gap
///     slivers, shadow edges, tray ridges, glare streaks;
///   - sprawling webs (tiny fill ratio of their bounding box): connected
///     gap networks between clustered pills;
///   - rings (large hole area relative to own area): the illumination-
///     flattening halo that surrounds a bright pill cluster on a dark tray
///     encircles the pills, so its "holes" ARE the pills.
void dropJunkComponents(cv::Mat& mask, double rMin) {
    if (cv::countNonZero(mask) == 0) return;
    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, 5);
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids,
                                             8, CV_32S);
    // Per-component max inscribed radius in one pass.
    std::vector<float> compMax(static_cast<size_t>(n), 0.f);
    for (int y = 0; y < labels.rows; ++y) {
        const int* l = labels.ptr<int>(y);
        const float* d = dist.ptr<float>(y);
        for (int x = 0; x < labels.cols; ++x)
            if (l[x] > 0) compMax[l[x]] = std::max(compMax[l[x]], d[x]);
    }
    std::vector<uchar> drop(static_cast<size_t>(n), 0);
    bool any = false;
    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        double extent = static_cast<double>(area) / std::max(w * h, 1);
        bool kill = compMax[i] < rMin || extent < 0.2;
        if (!kill) {
            int x0 = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y0 = stats.at<int>(i, cv::CC_STAT_TOP);
            cv::Mat comp = (labels(cv::Rect(x0, y0, w, h)) == i);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(comp, contours, cv::RETR_EXTERNAL,
                             cv::CHAIN_APPROX_SIMPLE);
            cv::Mat filled = cv::Mat::zeros(comp.size(), CV_8U);
            cv::drawContours(filled, contours, -1, 255, cv::FILLED);
            int holeArea = cv::countNonZero(filled) - area;
            kill = holeArea > 0.35 * area;
        }
        if (kill) {
            drop[i] = 1;
            any = true;
        }
    }
    if (!any) return;
    for (int y = 0; y < labels.rows; ++y) {
        const int* l = labels.ptr<int>(y);
        uchar* m = mask.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; ++x)
            if (l[x] > 0 && drop[l[x]]) m[x] = 0;
    }
}

/// Foreground = pixels deviating from the flattened tray level in luminance,
/// saturation, or rim-shading gradient. The luminance cue is split BY
/// POLARITY (lighter-than-tray vs darker-than-tray) and each side is
/// junk-filtered separately before the union: near bright pills the
/// flattened "tray level" is dragged upward, so the dark gaps between
/// touching bright pills read as deviation too — as slivers/webs/rings in
/// the opposite-polarity mask. Filtering per polarity removes them; a
/// single unsigned mask would glue the cluster into one blob (found on real
/// footage of white pills on a dark tray).
cv::Mat binaryMask(const cv::Mat& bgr, const cv::Mat& flat, const Params& p) {
    const cv::Size blurK(p.blurKernel, p.blurKernel);
    const double rMin =
        0.5 * std::sqrt(p.minAreaFraction * flat.total() / CV_PI);

    // Cue 1: |luminance - tray|, split by polarity.
    cv::Mat dev, mask;
    cv::absdiff(flat, cv::Scalar(128), dev);
    cv::GaussianBlur(dev, dev, blurK, 0);
    cv::Mat flatBlur;
    cv::GaussianBlur(flat, flatBlur, blurK, 0);
    {
        cv::Mat thresholded;
        double otsu = cv::threshold(dev, thresholded, 0, 255,
                                    cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (otsu < p.minDeviation) {
            // No real luminance signal (e.g. empty tray): don't let Otsu
            // promote sensor noise to foreground.
            mask = cv::Mat::zeros(dev.size(), CV_8U);
        } else {
            cv::Mat lightSide, darkSide, mLight, mDark;
            cv::compare(flatBlur, 128, lightSide, cv::CMP_GE);
            cv::bitwise_not(lightSide, darkSide);
            cv::bitwise_and(thresholded, lightSide, mLight);
            cv::bitwise_and(thresholded, darkSide, mDark);
            cv::Mat kOpen = ellipseKernel(p.openKernel);
            cv::morphologyEx(mLight, mLight, cv::MORPH_OPEN, kOpen);
            cv::morphologyEx(mDark, mDark, cv::MORPH_OPEN, kOpen);
            dropJunkComponents(mLight, rMin);
            dropJunkComponents(mDark, rMin);
            cv::bitwise_or(mLight, mDark, mask);
        }
    }

    // Cue 2: saturation — colored pills whose luminance matches the tray.
    {
        cv::Mat hsv, sat;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        cv::extractChannel(hsv, sat, 1);
        cv::GaussianBlur(sat, sat, blurK, 0);
        cv::Mat satMask;
        double otsu = cv::threshold(sat, satMask, 0, 255,
                                    cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (otsu >= p.minSaturation) {
            dropJunkComponents(satMask, rMin);
            cv::bitwise_or(mask, satMask, mask);
        }
    }

    // Cue 3: rim shading — a pill matching the tray in both brightness and
    // color still shows a shaded rim because it is convex. Strong Sobel
    // edges are closed into rings and filled.
    {
        cv::Mat fblur, gx, gy, mag, grad;
        cv::GaussianBlur(flat, fblur, blurK, 0);
        cv::Sobel(fblur, gx, CV_32F, 1, 0, 3);
        cv::Sobel(fblur, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, mag);
        cv::convertScaleAbs(mag, grad);
        cv::Mat edges;
        double otsu = cv::threshold(grad, edges, 0, 255,
                                    cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (otsu >= p.minGradient) {
            cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
                             ellipseKernel(p.gradCloseKernel));
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(edges, contours, cv::RETR_EXTERNAL,
                             cv::CHAIN_APPROX_SIMPLE);
            cv::Mat filled = cv::Mat::zeros(edges.size(), CV_8U);
            cv::drawContours(filled, contours, -1, 255, cv::FILLED);
            // A filled edge-ring can only be a rescued *single* pill. When
            // many pills touch, their rims connect into one ring and the
            // fill swallows the background gaps between them — discard any
            // filled component larger than the max pill area so this cue can
            // never glue a cluster together (found on real video).
            const double maxFill = p.maxAreaFraction * filled.total();
            cv::Mat labels, stats, centroids;
            int nFill = cv::connectedComponentsWithStats(
                filled, labels, stats, centroids, 8, CV_32S);
            std::vector<uchar> keep(static_cast<size_t>(nFill), 255);
            bool anyDropped = false;
            for (int i = 1; i < nFill; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > maxFill) {
                    keep[i] = 0;
                    anyDropped = true;
                }
            }
            if (anyDropped) {
                for (int y = 0; y < filled.rows; ++y) {
                    const int* l = labels.ptr<int>(y);
                    uchar* f = filled.ptr<uchar>(y);
                    for (int x = 0; x < filled.cols; ++x)
                        if (l[x] > 0 && !keep[l[x]]) f[x] = 0;
                }
            }
            cv::bitwise_or(mask, filled, mask);
        }
    }

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, ellipseKernel(p.openKernel));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, ellipseKernel(p.closeKernel));
    return mask;
}

/// Marker source: the mask minus strong thin dark valleys (contact shadows
/// between touching pills), so side-by-side pills seed separate markers.
/// The depth required is PROPORTIONAL to local brightness so contacts are
/// found on dark pills too, while shallow score-line grooves (which run
/// ~15-25% of pill brightness vs ~35-50% for contact shadows, measured on
/// real footage) are spared everywhere.
cv::Mat markerMask(const cv::Mat& flat, const cv::Mat& mask, const Params& p) {
    if (!p.valleyAssist) return mask.clone();
    cv::Mat blackhat, flatBlur;
    cv::morphologyEx(flat, blackhat, cv::MORPH_BLACKHAT,
                     ellipseKernel(p.valleyKernel));
    cv::GaussianBlur(flat, flatBlur, cv::Size(p.blurKernel, p.blurKernel), 0);
    cv::Mat required;
    flatBlur.convertTo(required, CV_8U, p.valleyRatio, 0.0);
    cv::max(required, static_cast<double>(p.valleyFloor), required);
    cv::Mat valley, out;
    cv::compare(blackhat, required, valley, cv::CMP_GE);
    cv::bitwise_not(valley, valley);
    cv::bitwise_and(mask, valley, out);
    return out;
}

/// Watershed with markers from an adaptively thresholded distance transform.
/// `mask` is modified in place by the thin-structure filter.
/// Returns the marker label image (background = 1, pills = 2..nMarkers).
cv::Mat splitTouching(const cv::Mat& bgr, cv::Mat& mask, cv::Mat& markerSrc,
                      const Params& p, int& nMarkers) {
    const double rMin =
        0.5 * std::sqrt(p.minAreaFraction * mask.total() / CV_PI);

    // Thin-structure filter: a component whose max inscribed radius is below
    // the minimum pill radius is line-like junk (shadow edges, tray ridges,
    // glare streaks) — erase it so it cannot swallow a neighbor's region.
    cv::Mat fullDist;
    cv::distanceTransform(mask, fullDist, cv::DIST_L2, 5);
    {
        cv::Mat comp;
        int nComp = cv::connectedComponents(mask, comp, 8, CV_32S);
        std::vector<float> compMax(static_cast<size_t>(nComp), 0.f);
        for (int y = 0; y < comp.rows; ++y) {
            const int* c = comp.ptr<int>(y);
            const float* d = fullDist.ptr<float>(y);
            for (int x = 0; x < comp.cols; ++x)
                if (c[x] > 0) compMax[c[x]] = std::max(compMax[c[x]], d[x]);
        }
        for (int y = 0; y < comp.rows; ++y) {
            const int* c = comp.ptr<int>(y);
            uchar* m = mask.ptr<uchar>(y);
            uchar* s = markerSrc.ptr<uchar>(y);
            for (int x = 0; x < comp.cols; ++x)
                if (c[x] > 0 && compMax[c[x]] < rMin) m[x] = s[x] = 0;
        }
    }

    // Distance transform of the (valley-subtracted) marker source, smoothed
    // to remove discretization bumps that would split single pills.
    cv::Mat dist;
    cv::distanceTransform(markerSrc, dist, cv::DIST_L2, 5);
    cv::GaussianBlur(dist, dist, cv::Size(5, 5), 0);

    // Adaptive threshold: compare each pixel to the max distance within a
    // local window (rect kernel = separable max filter, fast at this size).
    int win = makeOdd(p.markerWindowFraction * p.maxDimension);
    cv::Mat localMax;
    cv::dilate(dist, localMax,
               cv::getStructuringElement(cv::MORPH_RECT, cv::Size(win, win)));
    cv::Mat sureFg;
    {
        cv::Mat geLocal, gtMin;
        cv::compare(dist, localMax * p.distRatio, geLocal, cv::CMP_GE);
        cv::compare(dist, rMin, gtMin, cv::CMP_GT);
        cv::bitwise_and(geLocal, gtMin, sureFg);
    }

    // Safety net: every surviving mask component gets at least one marker,
    // so a pill can never be dropped at the marker stage (e.g. a small pill
    // whose window is dominated by a much larger neighbor's peak).
    {
        cv::Mat comp;
        int nComp = cv::connectedComponents(mask, comp, 8, CV_32S);
        std::vector<bool> hasMarker(static_cast<size_t>(nComp), false);
        std::vector<float> bestVal(static_cast<size_t>(nComp), 0.f);
        std::vector<cv::Point> bestAt(static_cast<size_t>(nComp));
        for (int y = 0; y < comp.rows; ++y) {
            const int* c = comp.ptr<int>(y);
            const uchar* s = sureFg.ptr<uchar>(y);
            const float* d = fullDist.ptr<float>(y);
            for (int x = 0; x < comp.cols; ++x) {
                if (c[x] <= 0) continue;
                if (s[x]) hasMarker[c[x]] = true;
                if (d[x] > bestVal[c[x]]) {
                    bestVal[c[x]] = d[x];
                    bestAt[c[x]] = cv::Point(x, y);
                }
            }
        }
        for (int i = 1; i < nComp; ++i)
            if (!hasMarker[i] && bestVal[i] > 0.f)
                cv::circle(sureFg, bestAt[i], 2, 255, cv::FILLED);
    }

    // Standard marker bookkeeping: background = 1, unknown = 0, seeds = 2+.
    cv::Mat markers;
    nMarkers = cv::connectedComponents(sureFg, markers, 8, CV_32S);
    for (int y = 0; y < markers.rows; ++y) {
        int* mk = markers.ptr<int>(y);
        const uchar* m = mask.ptr<uchar>(y);
        const uchar* s = sureFg.ptr<uchar>(y);
        for (int x = 0; x < markers.cols; ++x) {
            mk[x] += 1;
            if (m[x] && !s[x]) mk[x] = 0;
        }
    }
    // Flood on the color frame so region boundaries snap to intensity edges
    // (e.g. the shading seam between touching pills).
    cv::watershed(bgr, markers);
    return markers;
}

}  // namespace

FrameResult detectPills(const cv::Mat& bgr, const Params& p) {
    FrameResult result;
    if (bgr.empty()) return result;

    // Downscale to the working resolution.
    double scale = std::min(
        1.0, static_cast<double>(p.maxDimension) / std::max(bgr.cols, bgr.rows));
    cv::Mat work;
    if (scale < 1.0) {
        cv::resize(bgr, work, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        work = bgr;
    }
    result.workSize = work.size();

    cv::Mat gray, flat;
    cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
    flat = flattenIllumination(gray, p);

    cv::Mat mask = binaryMask(work, flat, p);
    cv::Mat markerSrc = markerMask(flat, mask, p);

    int nMarkers = 0;
    cv::Mat markers = splitTouching(work, mask, markerSrc, p, nMarkers);

    // Per-region bounding boxes in one pass, so contour extraction only
    // touches each region's ROI instead of the full frame per region.
    std::vector<cv::Rect> boxes(static_cast<size_t>(nMarkers) + 2,
                                cv::Rect(0, 0, 0, 0));
    for (int y = 0; y < markers.rows; ++y) {
        const int* mk = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x) {
            int label = mk[x];
            if (label < 2) continue;  // 1 = background, -1 = watershed line
            cv::Rect& b = boxes[label];
            if (b.width == 0) {
                b = cv::Rect(x, y, 1, 1);
            } else {
                b |= cv::Rect(x, y, 1, 1);
            }
        }
    }

    const double totalPx = static_cast<double>(work.total());
    const double minArea = p.minAreaFraction * totalPx;
    const double maxArea = p.maxAreaFraction * totalPx;
    const float invW = 1.0f / static_cast<float>(work.cols);
    const float invH = 1.0f / static_cast<float>(work.rows);

    for (int label = 2; label <= nMarkers; ++label) {
        const cv::Rect& box = boxes[label];
        if (box.width == 0) continue;

        cv::Mat region = (markers(box) == label);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(region, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE, box.tl());
        if (contours.empty()) continue;
        auto& contour = *std::max_element(
            contours.begin(), contours.end(),
            [](const auto& a, const auto& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });

        double area = cv::contourArea(contour);
        if (area < minArea || area > maxArea) continue;

        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hullArea = cv::contourArea(hull);
        if (hullArea <= 0 || area / hullArea < p.minSolidity) continue;

        cv::RotatedRect rect = cv::minAreaRect(contour);
        float lo = std::min(rect.size.width, rect.size.height);
        float hi = std::max(rect.size.width, rect.size.height);
        if (lo <= 0 || hi / std::max(lo, 1e-3f) > p.maxAspect) continue;

        Pill pill;
        pill.areaFraction = area / totalPx;
        pill.center = cv::Point2f(rect.center.x * invW, rect.center.y * invH);
        pill.contour.reserve(contour.size());
        for (const cv::Point& pt : contour)
            pill.contour.emplace_back(pt.x * invW, pt.y * invH);
        result.pills.push_back(std::move(pill));
    }
    return result;
}

}  // namespace pillcount
