"""Reference implementation of the pill-detection pipeline.

This mirrors, step for step, the C++ pipeline in the iOS app
(PillCount/Detection/OpenCVWrapper.mm). Used to validate the algorithm and
tune parameters against test images before/without touching the Swift code.

Pipeline overview
-----------------
1.  Downscale to a working resolution (max_dim).
2.  Illumination flattening: divide grayscale by a heavily blurred copy so
    the tray maps to a uniform ~128 regardless of lighting gradients/shadows.
3.  Foreground mask = |flattened - 128| over an Otsu threshold (catches pills
    both lighter AND darker than the tray in the same frame), OR'd with a
    saturation mask (catches colored pills whose luminance matches the tray).
4.  Morphological open/close to clean noise.
5.  Watershed markers from the distance transform, thresholded adaptively
    against the local neighborhood max (separates touching pills of mixed
    sizes). Dark "contact seams" between touching pills are subtracted before
    the distance transform so side-by-side pills seed separate markers.
6.  Watershed on the color frame; per-region contour extraction; filter by
    area / solidity / aspect ratio.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Params:
    # Working resolution: frames are downscaled so max(w, h) <= this.
    max_dim: int = 640
    # Illumination flattening: background blur kernel as fraction of max_dim.
    bg_blur_frac: float = 0.25
    # Pre-threshold blur kernel (odd).
    blur_ksize: int = 5
    # Deviation floor: if Otsu picks a threshold below this, the frame is
    # treated as having no real foreground (prevents phantom detections on an
    # empty tray, where Otsu would happily threshold sensor noise).
    min_dev: int = 8
    # Saturation mask is only trusted when Otsu on the saturation channel
    # exceeds this (i.e. there is genuine color signal in the scene).
    min_sat: int = 40
    # Gradient (rim-shading) cue: pills that match the tray in brightness and
    # color still show edge shading because they are convex. Sobel magnitude
    # below this floor is ignored (prevents empty-tray phantom edges).
    min_grad: int = 18
    grad_close_ksize: int = 7
    # Area limits as a fraction of the working-frame pixel count.
    min_area_frac: float = 0.0005
    max_area_frac: float = 0.05
    # Shape filters applied to final per-pill contours.
    min_solidity: float = 0.70
    max_aspect: float = 4.5
    # Watershed markers: a pixel is "sure foreground" when its distance value
    # is at least dist_ratio * (max distance within marker_window px).
    dist_ratio: float = 0.45
    marker_window_frac: float = 0.09   # window = frac * max_dim, odd
    # Valley assist: subtract thin dark contact seams (morphological
    # black-hat) from the mask before the distance transform, so touching
    # pills seed separate markers. The threshold is PROPORTIONAL to local
    # brightness: contact shadows between convex pills run ~35-50% of the
    # pill's brightness while score-line grooves run ~15-25% (measured on
    # real footage of scored white caplets), so this cuts contacts without
    # splitting scored tablets — on both bright and dark pills. Disable if
    # calibration mode shows score-line over-splitting on your stock.
    valley_assist: bool = True
    valley_ratio: float = 0.35         # black-hat depth / local flat level
    valley_floor: int = 15             # absolute minimum depth (noise gate)
    valley_ksize: int = 7              # black-hat kernel (thin structures)
    # Morphology kernel sizes.
    open_ksize: int = 3
    close_ksize: int = 3


@dataclass
class Detection:
    contour: np.ndarray  # Nx2 int points in working-frame coords
    center: tuple
    area: float


@dataclass
class Result:
    detections: list = field(default_factory=list)
    work_size: tuple = (0, 0)  # (w, h) of working frame

    @property
    def count(self):
        return len(self.detections)


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _flatten_illumination(gray, p: Params):
    """Divide by a heavily blurred copy: tray -> ~128, independent of lighting
    gradients and soft shadows."""
    k = _odd(max(gray.shape) * p.bg_blur_frac)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.divide(gray, bg, scale=128)


def _drop_junk_components(mask, r_min):
    """Remove non-pill-like components. Pills (and clusters of touching
    pills) are solid, reasonably thick blobs; the artifacts this rejects are:
      - thin structures (max inscribed radius < min pill radius): gap
        slivers, shadow edges, tray ridges, glare streaks;
      - sprawling webs (tiny fill ratio of their bounding box): connected
        gap networks between clustered pills;
      - rings (large hole area relative to own area): the illumination-
        flattening halo that surrounds a bright pill cluster on a dark tray
        encircles the pills, so its "holes" ARE the pills."""
    if not mask.any():
        return mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bbox_area = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        sel = labels == i
        drop = dist[sel].max() < r_min or area / max(bbox_area, 1) < 0.2
        if not drop:
            comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(comp)
            cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
            hole_area = int(filled.sum()) - area
            drop = hole_area > 0.35 * area
        if drop:
            mask[sel] = 0
    return mask


def _binary_mask(bgr, flat, p: Params):
    """Foreground = pixels deviating from the flattened tray level, in
    luminance or saturation. The luminance cue is split BY POLARITY
    (lighter-than-tray vs darker-than-tray) and each side is thin-filtered
    separately before the union: near bright pills the flattened "tray level"
    is dragged upward, so dark gaps between touching bright pills read as
    deviation too — as slivers in the opposite-polarity mask. Filtering per
    polarity removes them; a single unsigned mask would glue the cluster."""
    r_min = 0.5 * np.sqrt(p.min_area_frac * flat.size / np.pi)
    dev = cv2.absdiff(flat, 128)
    dev = cv2.GaussianBlur(dev, (p.blur_ksize,) * 2, 0)
    flat_blur = cv2.GaussianBlur(flat, (p.blur_ksize,) * 2, 0)
    otsu_t, _ = cv2.threshold(dev, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if otsu_t < p.min_dev:
        mask = np.zeros_like(dev)      # empty tray: no real signal
    else:
        _, m_dev = cv2.threshold(dev, otsu_t, 255, cv2.THRESH_BINARY)
        m_light = cv2.bitwise_and(m_dev, (flat_blur >= 128).astype(np.uint8) * 255)
        m_dark = cv2.bitwise_and(m_dev, (flat_blur < 128).astype(np.uint8) * 255)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (p.open_ksize,) * 2)
        m_light = cv2.morphologyEx(m_light, cv2.MORPH_OPEN, k_open)
        m_dark = cv2.morphologyEx(m_dark, cv2.MORPH_OPEN, k_open)
        mask = cv2.bitwise_or(_drop_junk_components(m_light, r_min),
                              _drop_junk_components(m_dark, r_min))

    # Colored pills whose luminance matches the tray still pop in saturation.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (p.blur_ksize,) * 2, 0)
    sat_t, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if sat_t >= p.min_sat:
        _, m_sat = cv2.threshold(sat, sat_t, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, _drop_junk_components(m_sat, r_min))

    # Rim-shading cue: a pill whose face matches the tray in both brightness
    # and color still shows a shaded rim (it is convex). Take strong Sobel
    # edges of the flattened image, close them into rings, and fill.
    fblur = cv2.GaussianBlur(flat, (p.blur_ksize,) * 2, 0)
    gx = cv2.Sobel(fblur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(fblur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    grad_t, _ = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if grad_t >= p.min_grad:
        _, edges = cv2.threshold(grad, grad_t, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (p.grad_close_ksize,) * 2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(edges)
        cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
        # A filled edge-ring can only be a rescued *single* pill. When many
        # pills touch, their rims connect into one ring and the fill swallows
        # the background gaps between them — discard any filled component
        # larger than the max pill area so this cue can never glue a cluster
        # together (found on real video, see docs/LIMITATIONS.md).
        max_fill = p.max_area_frac * filled.size
        n_f, fl, fstats, _ = cv2.connectedComponentsWithStats(filled, 8)
        for i in range(1, n_f):
            if fstats[i, cv2.CC_STAT_AREA] > max_fill:
                filled[fl == i] = 0
        mask = cv2.bitwise_or(mask, filled)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_ksize,) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_ksize,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def _marker_mask(flat, mask, p: Params):
    """Mask used only for seeding markers: the foreground mask minus strong
    thin dark valleys (contact shadows between touching pills). The depth
    required scales with local brightness so contacts are found on dark
    pills too, while shallow score-line grooves are spared everywhere."""
    if not p.valley_assist:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.valley_ksize,) * 2)
    blackhat = cv2.morphologyEx(flat, cv2.MORPH_BLACKHAT, k)
    flat_blur = cv2.GaussianBlur(flat, (p.blur_ksize,) * 2, 0)
    required = np.maximum(
        p.valley_floor,
        (p.valley_ratio * flat_blur.astype(np.float32)).astype(np.uint8))
    valley = (blackhat >= required).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, cv2.bitwise_not(valley))


def _split_touching(bgr, mask, marker_src, p: Params):
    """Watershed with markers from an adaptively thresholded distance
    transform: a pixel seeds foreground when its distance value is close to
    the max within a local window. Handles touching pills of mixed sizes
    (a global or per-component threshold does not)."""
    r_min = 0.5 * np.sqrt(p.min_area_frac * mask.size / np.pi)

    # Thin-structure filter: a mask component whose max inscribed radius is
    # below the minimum pill radius is line-like junk (shadow edges, tray
    # ridges, glare streaks) — erase it so it cannot swallow a neighboring
    # pill's watershed region.
    full_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    n_c, comp = cv2.connectedComponents(mask, 8)
    for i in range(1, n_c):
        sel = comp == i
        if full_dist[sel].max() < r_min:
            mask[sel] = 0
            marker_src[sel] = 0

    dist = cv2.distanceTransform(marker_src, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)

    # Rect kernel: separable max filter, far faster than ellipse at this size.
    win = _odd(p.marker_window_frac * p.max_dim)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    local_max = cv2.dilate(dist, kernel)
    sure_fg = ((dist >= p.dist_ratio * local_max) & (dist > r_min)
               ).astype(np.uint8) * 255

    # Safety net: every mask component gets at least one marker, so a pill
    # can never be dropped by the marker stage (e.g. a small pill next to a
    # much larger one whose distance peak dominates the window).
    n_c, comp = cv2.connectedComponents(mask, 8)
    for i in range(1, n_c):
        sel = comp == i
        if not sure_fg[sel].any():
            d = np.where(sel, full_dist, 0)
            if d.max() <= 0:
                continue
            y, x = np.unravel_index(np.argmax(d), d.shape)
            cv2.circle(sure_fg, (int(x), int(y)), 2, 255, -1)

    n_m, markers = cv2.connectedComponents(sure_fg, 8)
    markers = markers + 1              # background label becomes 1
    markers[(mask > 0) & (sure_fg == 0)] = 0   # unknown region
    cv2.watershed(bgr, markers)
    return markers, n_m


def detect(bgr, p: Params = None) -> Result:
    p = p or Params()
    h, w = bgr.shape[:2]
    scale = min(1.0, p.max_dim / max(h, w))
    work = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr.copy()

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    flat = _flatten_illumination(gray, p)
    mask = _binary_mask(work, flat, p)
    marker_src = _marker_mask(flat, mask, p)
    markers, n_markers = _split_touching(work, mask, marker_src, p)

    total_px = work.shape[0] * work.shape[1]
    min_a, max_a = p.min_area_frac * total_px, p.max_area_frac * total_px
    res = Result(work_size=(work.shape[1], work.shape[0]))

    for m in range(2, n_markers + 1):     # label 1 = background
        region = (markers == m).astype(np.uint8)
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_a or area > max_a:
            continue
        hull = cv2.convexHull(c)
        hull_a = cv2.contourArea(hull)
        if hull_a <= 0 or area / hull_a < p.min_solidity:
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) <= 0 or max(rw, rh) / max(min(rw, rh), 1e-3) > p.max_aspect:
            continue
        res.detections.append(Detection(c.reshape(-1, 2), (cx, cy), area))
    return res


def annotate(bgr, res: Result):
    out = bgr.copy()
    for d in res.detections:
        cv2.drawContours(out, [d.contour.reshape(-1, 1, 2)], -1, (0, 255, 80), 2)
        cv2.circle(out, (int(d.center[0]), int(d.center[1])), 3, (0, 100, 255), -1)
    cv2.putText(out, f"count: {res.count}", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 80), 2)
    return out
