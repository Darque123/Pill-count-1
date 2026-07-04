"""Synthetic accuracy tests for the pill-detection pipeline.

Renders pharmacy-tray-like scenes with known ground-truth counts, including
the known hard cases, and checks the pipeline count. Run: python3 test_synthetic.py
"""
import cv2
import numpy as np
from pipeline import detect, annotate, Params

rng = np.random.default_rng(7)
W, H = 1280, 960


def tray(color=(200, 205, 210), gradient=0.0):
    img = np.full((H, W, 3), color, np.uint8)
    if gradient:
        gy = np.linspace(1 - gradient, 1 + gradient, H)[:, None, None]
        img = np.clip(img * gy, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def draw_round(img, c, r, color, highlight=True):
    cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
    # rim shading — real pills are convex, so the edge is always darker
    cv2.circle(img, c, r, tuple(int(v * 0.72) for v in color), 3, cv2.LINE_AA)
    cv2.circle(img, c, r - 3, tuple(int(v * 0.86) for v in color), 3, cv2.LINE_AA)
    if highlight:
        hc = (c[0] - r // 3, c[1] - r // 3)
        cv2.circle(img, hc, max(2, r // 4),
                   tuple(min(255, int(v * 1.25) + 20) for v in color), -1, cv2.LINE_AA)


def draw_caplet(img, c, half_len, r, angle_deg, color):
    a = np.deg2rad(angle_deg)
    d = np.array([np.cos(a), np.sin(a)]) * (half_len - r)
    p1 = (int(c[0] - d[0]), int(c[1] - d[1]))
    p2 = (int(c[0] + d[0]), int(c[1] + d[1]))
    cv2.line(img, p1, p2, color, r * 2, cv2.LINE_AA)
    cv2.circle(img, p1, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, r, color, -1, cv2.LINE_AA)


def scene_scattered():
    """12 well-separated round pills, mixed colors."""
    img = tray()
    colors = [(60, 60, 230), (230, 150, 60), (250, 250, 252), (40, 160, 240)]
    pts, n = [], 12
    while len(pts) < n:
        p = (rng.integers(80, W - 80), rng.integers(80, H - 80))
        if all((p[0]-q[0])**2 + (p[1]-q[1])**2 > 130**2 for q in pts):
            pts.append(p)
    for i, p in enumerate(pts):
        draw_round(img, (int(p[0]), int(p[1])), int(rng.integers(28, 40)),
                   colors[i % len(colors)])
    return img, n


def scene_touching_cluster():
    """7 same-size round pills packed hex-style (all touching) + 3 loose."""
    img = tray()
    r = 34
    cx, cy = W // 2, H // 2
    centers = [(cx, cy)]
    for k in range(6):
        a = k * np.pi / 3
        centers.append((int(cx + 2 * r * 0.98 * np.cos(a)),
                        int(cy + 2 * r * 0.98 * np.sin(a))))
    for c in centers:
        draw_round(img, c, r, (70, 70, 225))
    loose = [(200, 200), (W - 220, 220), (240, H - 200)]
    for c in loose:
        draw_round(img, c, r, (70, 70, 225))
    return img, len(centers) + len(loose)


def caplet_pair(img, c, half_len, r, angle_deg, color):
    """Two parallel caplets touching side by side, with the thin contact
    shadow that real convex pills cast where they meet."""
    a = np.deg2rad(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    perp = np.array([-np.sin(a), np.cos(a)])
    c1 = np.array(c, float) - perp * r
    c2 = np.array(c, float) + perp * r
    draw_caplet(img, tuple(c1.astype(int)), half_len, r, angle_deg, color)
    draw_caplet(img, tuple(c2.astype(int)), half_len, r, angle_deg, color)
    seam_half = half_len - r
    p1 = (np.array(c, float) - d * seam_half).astype(int)
    p2 = (np.array(c, float) + d * seam_half).astype(int)
    seam_color = tuple(int(v * 0.55) for v in color)
    cv2.line(img, tuple(p1), tuple(p2), seam_color, 3, cv2.LINE_AA)


def scene_touching_pairs_caplets():
    """Oblong caplets: two side-by-side touching pairs + singles."""
    img = tray()
    color = (90, 200, 250)
    caplet_pair(img, (380, 320), 70, 24, 20, color)
    caplet_pair(img, (870, 640), 70, 24, -35, color)
    draw_caplet(img, (900, 250), 70, 24, 80, (250, 250, 252))
    draw_caplet(img, (300, 700), 70, 24, 5, (60, 60, 230))
    return img, 6


def scene_white_on_light():
    """White pills on a light tray with an illumination gradient — the
    low-contrast case."""
    img = tray(color=(215, 218, 222), gradient=0.18)
    n = 9
    pts = []
    while len(pts) < n:
        p = (int(rng.integers(90, W - 90)), int(rng.integers(90, H - 90)))
        if all((p[0]-q[0])**2 + (p[1]-q[1])**2 > 140**2 for q in pts):
            pts.append(p)
    for p in pts:
        draw_round(img, p, 34, (243, 244, 246), highlight=False)
    return img, n


def scene_glare_shadow():
    """Round pills with specular glare blobs and a cast shadow band."""
    img = tray()
    # shadow band with a realistic soft penumbra
    shade = np.ones((H, W), np.float32)
    shade[300:520, :] = 0.72
    shade = cv2.GaussianBlur(shade, (31, 31), 0)
    img = np.clip(img.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
    pts = [(220, 400), (500, 380), (760, 430), (1020, 400),
           (320, 720), (640, 700), (960, 740), (620, 180)]
    for p in pts:
        draw_round(img, p, 36, (50, 170, 60))
        # harsh specular glare
        cv2.circle(img, (p[0] - 10, p[1] - 12), 9, (255, 255, 255), -1, cv2.LINE_AA)
    return img, len(pts)


def scene_mixed_sizes_touching():
    """Small and large pills, one touching pair of each size."""
    img = tray(color=(150, 155, 160))
    draw_round(img, (300, 300), 22, (250, 250, 252))
    draw_round(img, (300 + 43, 300), 22, (250, 250, 252))   # small touching pair
    draw_round(img, (800, 500), 48, (60, 60, 230))
    draw_round(img, (800 + 94, 500), 48, (60, 60, 230))     # large touching pair
    draw_round(img, (400, 720), 22, (250, 250, 252))
    draw_round(img, (1000, 200), 48, (60, 60, 230))
    return img, 6


def scene_empty_tray():
    """No pills: the count must be 0, not sensor-noise phantoms."""
    return tray(gradient=0.12), 0


def scene_dark_tray():
    """White and colored pills on a dark tray."""
    img = tray(color=(60, 62, 66))
    pts = [(250, 250), (600, 300), (950, 260), (350, 600), (750, 650), (1050, 550)]
    for i, p in enumerate(pts):
        draw_round(img, p, 34, (250, 250, 252) if i % 2 else (50, 60, 200))
    return img, len(pts)


SCENES = [
    ("empty_tray", scene_empty_tray),
    ("dark_tray", scene_dark_tray),
    ("scattered", scene_scattered),
    ("touching_cluster", scene_touching_cluster),
    ("touching_caplets", scene_touching_pairs_caplets),
    ("white_on_light", scene_white_on_light),
    ("glare_shadow", scene_glare_shadow),
    ("mixed_sizes_touching", scene_mixed_sizes_touching),
]


def dump_fixtures(outdir):
    """Write scene PNGs + a truth manifest for the C++ harness
    (tools/cpp_test), which regression-tests the shipping C++ pipeline
    against the same scenes."""
    import os
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "manifest.csv"), "w") as f:
        for name, fn in SCENES:
            img, truth = fn()
            cv2.imwrite(os.path.join(outdir, f"{name}.png"), img)
            f.write(f"{name}.png,{truth}\n")
    print(f"wrote {len(SCENES)} fixtures to {outdir}")


def main():
    import os, sys, time
    if len(sys.argv) > 2 and sys.argv[1] == "--dump":
        dump_fixtures(sys.argv[2])
        return
    outdir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    p = Params()
    failures = 0
    for name, fn in SCENES:
        img, truth = fn()
        t0 = time.perf_counter()
        res = detect(img, p)
        ms = (time.perf_counter() - t0) * 1000
        ok = res.count == truth
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name:24s} truth={truth:3d} "
              f"got={res.count:3d}  ({ms:.1f} ms)")
        scale = res.work_size[0] / img.shape[1]
        work = cv2.resize(img, res.work_size)
        cv2.imwrite(os.path.join(outdir, f"{name}.png"), annotate(work, res))
    print("all passed" if failures == 0 else f"{failures} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
