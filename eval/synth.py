"""SYNTHETIC test scenes with known crack widths and a known ArUco scale.

Everything here is generated: procedural concrete-like texture, a crack polyline rendered
with exact anti-aliased coverage at a known width (mm), a printed ArUco marker of known
size, then a random camera homography, optical blur, sensor noise and JPEG compression.
Use it to measure scale and width error; it says nothing about detection on real photos.
"""
from __future__ import annotations

import math

import cv2 as cv
import numpy as np

SS = 4  # supersampling factor for coverage-exact rendering


def texture(h, w, rng):
    base = np.zeros((h, w), np.float32)
    for octave, amp in [(8, 18), (32, 10), (128, 6), (512, 4)]:
        n = rng.normal(0, 1, (max(2, h // octave), max(2, w // octave))).astype(np.float32)
        base += cv.resize(n, (w, h), interpolation=cv.INTER_CUBIC) * amp
    base += rng.normal(0, 4, (h, w)).astype(np.float32)
    img = 165 + base
    # pores: small dark blobs (distractors)
    for _ in range(int(h * w / 9000)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = rng.uniform(0.6, 2.5)
        cv.circle(img, (int(x), int(y)), int(math.ceil(r)), float(rng.uniform(90, 140)), -1, cv.LINE_AA)
    return np.clip(img, 0, 255)


def crack_path(w_mm_world, h_mm_world, rng, ppm):
    """Random-walk polyline across the plane (world px coordinates at ppm px/mm)."""
    x, y = rng.uniform(0.28, 0.4) * w_mm_world, rng.uniform(0.3, 0.7) * h_mm_world
    ang = rng.uniform(-0.5, 0.5)
    pts = []
    L = rng.uniform(0.45, 0.6) * w_mm_world
    step = 0.8
    base = ang
    for _ in range(int(L / step)):
        pts.append((x * ppm, y * ppm))
        ang += rng.normal(0, 0.22) - 0.15 * (ang - base)     # jagged but mean-reverting
        base += rng.normal(0, 0.03)
        ang = float(np.clip(ang, -1.0, 1.0))
        x += step * math.cos(ang); y += step * math.sin(ang)
    return np.array(pts, np.float32)


_BG_CACHE = []


def real_backgrounds(n=24, size=(1600, 1100)):
    """Crack-free windows cut from TEST-split photos of the Özgenel segmentation set
    (real concrete texture, no crack pixels within 40 px). Grey-scale."""
    if _BG_CACHE:
        return _BG_CACHE
    from eval.common import seg_pairs, load_split
    test = set(load_split()["test"])
    for stem, ip, mp in seg_pairs():
        if stem not in test:
            continue
        img = cv.imread(ip, cv.IMREAD_GRAYSCALE)
        gt = cv.imread(mp, cv.IMREAD_GRAYSCALE)
        if gt.shape != img.shape:
            gt = cv.resize(gt, img.shape[::-1], interpolation=cv.INTER_NEAREST)
        bad = cv.dilate((gt > 127).astype(np.uint8), np.ones((81, 81), np.uint8))
        integ = cv.integral(bad)
        w, h = size
        H, W = img.shape
        best = None
        for y in range(0, H - h, 100):
            for x in range(0, W - w, 100):
                s = integ[y + h, x + w] - integ[y, x + w] - integ[y + h, x] + integ[y, x]
                if s == 0:
                    best = (x, y)
                    break
            if best:
                break
        if best:
            x, y = best
            _BG_CACHE.append((stem, img[y:y + h, x:x + w].astype(np.float32)))
        if len(_BG_CACHE) >= n:
            break
    return _BG_CACHE


def render_scene(rng, width_mm, cam_ppm, marker_mm=50.0, tilt_deg=20.0, blur=0.8, noise=3.0,
                 jpeg=88, world=(300, 200), background="real"):
    W, H = world
    ppm_w = max(cam_ppm * 1.6, 8.0)                    # world raster finer than the camera
    w_px, h_px = int(W * ppm_w), int(H * ppm_w)
    if background == "real":
        bgs = real_backgrounds()
        stem, bg = bgs[int(rng.integers(len(bgs)))]
        img = cv.resize(bg, (w_px, h_px), interpolation=cv.INTER_CUBIC)
    else:
        stem, img = "procedural", texture(h_px, w_px, rng)
    # crack: coverage from the exact distance to the centreline (cv.polylines thickness is
    # quantised to odd pixel counts, so it is NOT used for the width). Centreline drawn at
    # 2x on a crop around the path, distance transform -> analytic coverage of a w-wide band.
    pts = crack_path(W, H, rng, ppm_w)
    f = 2.0
    pad = int(width_mm * ppm_w * f) + 8
    x0, y0 = np.floor(pts.min(0) * f).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0) * f).astype(int) + pad
    line = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    cv.polylines(line, [((pts * f) - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)], False, 0, 1, cv.LINE_8)
    dist = cv.distanceTransform(line, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    cov_ss = np.clip(width_mm * ppm_w * f / 2 + 0.5 - dist, 0, 1).astype(np.float32)
    cov = np.zeros((h_px, w_px), np.float32)
    X0, Y0 = x0 // 2, y0 // 2
    cov_w = cv.resize(cov_ss, (cov_ss.shape[1] // 2, cov_ss.shape[0] // 2), interpolation=cv.INTER_AREA)
    ys, xs = slice(max(Y0, 0), min(Y0 + cov_w.shape[0], h_px)), slice(max(X0, 0), min(X0 + cov_w.shape[1], w_px))
    cov[ys, xs] = cov_w[ys.start - Y0:ys.stop - Y0, xs.start - X0:xs.stop - X0]
    depth = rng.uniform(0.55, 0.8)
    img = img * (1 - depth * cov)
    # marker with white quiet zone, top-left area
    mk_px = int(marker_mm * ppm_w)
    q = int(mk_px * 0.25)
    marker = cv.aruco.generateImageMarker(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50), 0, mk_px)
    mx, my = int(0.14 * w_px), int(0.17 * h_px)
    img[my - q:my + mk_px + q, mx - q:mx + mk_px + q] = 235
    img[my:my + mk_px, mx:mx + mk_px] = np.where(marker > 0, 235, 25)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # camera: homography with tilt, output at cam_ppm (centre)
    out_w, out_h = int(W * cam_ppm * 1.05), int(H * cam_ppm * 1.05)
    src = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], np.float32)
    k = cam_ppm / ppm_w
    t = math.tan(math.radians(tilt_deg)) * rng.choice([-1, 1])
    jit = rng.uniform(-0.04, 0.04, (4, 2)) * [out_w, out_h]
    dst = src * k + np.array([out_w * 0.025, out_h * 0.025])
    dst[0, 0] += t * 0.08 * out_w; dst[3, 0] += t * 0.08 * out_w        # keystone
    dst[1, 0] -= t * 0.08 * out_w; dst[2, 0] -= t * 0.08 * out_w
    dst[0, 1] += abs(t) * 0.05 * out_h; dst[1, 1] += abs(t) * 0.05 * out_h
    dst = (dst + jit * 0.3).astype(np.float32)
    Hm = cv.getPerspectiveTransform(src, dst)
    cam = cv.warpPerspective(img, Hm, (out_w, out_h), flags=cv.INTER_AREA, borderValue=128)
    # crop so the edge of the synthetic world never shows (a real wall has no such edge)
    cx0, cy0 = int(0.09 * out_w), int(0.09 * out_h)
    cam = cam[cy0:out_h - cy0, cx0:out_w - cx0]
    Hm = np.array([[1, 0, -cx0], [0, 1, -cy0], [0, 0, 1]], np.float64) @ Hm
    cam = cv.GaussianBlur(cam.astype(np.float32), (0, 0), blur)
    cam = cam + rng.normal(0, noise, cam.shape)
    cam = cv.cvtColor(np.clip(cam, 0, 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
    ok, buf = cv.imencode(".jpg", cam, [cv.IMWRITE_JPEG_QUALITY, jpeg])
    cam = cv.imdecode(buf, cv.IMREAD_COLOR)
    return {"image": cam, "background": stem, "width_mm": width_mm, "cam_ppm": cam_ppm, "marker_mm": marker_mm,
            "tilt_deg": tilt_deg, "blur": blur, "H_world_to_cam": Hm, "ppm_world": ppm_w,
            "path_world_px": pts}
