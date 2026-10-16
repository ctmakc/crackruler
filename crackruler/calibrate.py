"""Metric scale from an ArUco marker lying on the inspected surface (OpenCV 5 objdetect)."""
from __future__ import annotations

import cv2 as cv
import numpy as np

DEFAULT_DICT = cv.aruco.DICT_4X4_50
MAX_RECTIFIED_SIDE = 4096


def _detector(dict_id: int = DEFAULT_DICT) -> cv.aruco.ArucoDetector:
    params = cv.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    return cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(dict_id), params)


def detect_marker(bgr: np.ndarray, marker_id: int | None = None, dict_id: int = DEFAULT_DICT):
    """Return (corners[4,2] float32, id) of the largest matching marker, or (None, None)."""
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    corners, ids, _ = _detector(dict_id).detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None
    best, best_area, best_id = None, 0.0, None
    for c, i in zip(corners, ids.ravel()):
        if marker_id is not None and int(i) != marker_id:
            continue
        area = abs(cv.contourArea(c.reshape(-1, 2)))
        if area > best_area:
            best, best_area, best_id = c.reshape(4, 2).astype(np.float32), area, int(i)
    return best, best_id


def rectify(bgr: np.ndarray, corners: np.ndarray, marker_mm: float):
    """Warp the image so the marker plane is fronto-parallel at a uniform px/mm.

    px/mm is chosen to keep the marker's native pixel density (no invented detail).
    Returns dict with rectified image, homography H (src->rect), px_per_mm and an
    estimate of the relative scale uncertainty.
    """
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    px_per_mm = float(sides.mean() / marker_mm)
    s = marker_mm * px_per_mm
    dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float32)
    H = cv.getPerspectiveTransform(corners, dst)
    h, w = bgr.shape[:2]
    img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    warped = cv.perspectiveTransform(img_corners, H).reshape(-1, 2)
    x0, y0 = warped.min(0)
    x1, y1 = warped.max(0)
    # Clip extreme perspective: keep a window around the marker.
    cx, cy = s / 2, s / 2
    half = MAX_RECTIFIED_SIDE / 2
    x0, x1 = max(x0, cx - half), min(x1, cx + half)
    y0, y1 = max(y0, cy - half), min(y1, cy + half)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    H2 = T @ H
    out_w, out_h = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
    rect = cv.warpPerspective(bgr, H2, (out_w, out_h), flags=cv.INTER_LINEAR,
                              borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv.warpPerspective(np.full((h, w), 255, np.uint8), H2, (out_w, out_h),
                               flags=cv.INTER_NEAREST, borderValue=0)
    # crop to the valid (imaged) area so the evidence view carries no black wedges
    bx, by, bw, bh = cv.boundingRect(cv.findNonZero(valid))
    rect, valid = rect[by:by + bh, bx:bx + bw], valid[by:by + bh, bx:bx + bw]
    H2 = np.array([[1, 0, -bx], [0, 1, -by], [0, 0, 1]], np.float64) @ H2
    marker_rect = cv.perspectiveTransform(corners.reshape(-1, 1, 2), H2).reshape(-1, 2)
    # Sub-pixel corner error of ~0.25 px on each corner -> relative scale error.
    rel_unc = float(max(0.35 / sides.mean(), 0.005))
    skew = float(sides.std() / sides.mean())
    return {"image": rect, "valid": valid, "H": H2, "px_per_mm": px_per_mm,
            "rel_scale_unc": rel_unc, "marker_px": float(sides.mean()), "marker_skew": skew,
            "marker_rect": marker_rect}


def marker_mask(shape, corners: np.ndarray, pad_px: int = 6) -> np.ndarray:
    """Mask that covers the marker (its black border looks like a crack to ridge filters)."""
    m = np.zeros(shape[:2], np.uint8)
    c = corners.reshape(-1, 2)
    centre = c.mean(0)
    grown = centre + (c - centre) * 1.0 + np.sign(c - centre) * pad_px
    cv.fillConvexPoly(m, grown.astype(np.int32), 255)
    return m


def estimate_psf_sigma(gray: np.ndarray, marker_rect: np.ndarray) -> float | None:
    """Blur probe: the marker's outer black/white edge is a known sharp step.

    Averages profiles across the middle 60 % of each side, measures the 10-90 % rise and
    converts it to a Gaussian sigma (rise = 2.563 sigma). The value includes optics, motion,
    demosaicing, JPEG and our own rectification resampling - everything the crack profile sees.
    """
    c = marker_rect.reshape(4, 2).astype(np.float32)
    centre = c.mean(0)
    side = float(np.linalg.norm(c - np.roll(c, -1, 0), axis=1).mean())
    L = max(4.0, side / 14.0)          # stay inside the 1-cell black border (side / 6)
    t = np.arange(-L, L + 0.125, 0.25, dtype=np.float32)
    sigmas = []
    g = gray.astype(np.float32)
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        n = np.array([-d[1], d[0]], np.float32)
        if np.dot(n, centre - (a + b) / 2) > 0:
            n = -n                      # normal points outwards (black -> white)
        s = np.linspace(0.2, 0.8, 40, dtype=np.float32)
        base = a[None] + (b - a)[None] * s[:, None]
        mx = (base[:, 0:1] + n[0] * t[None]).astype(np.float32)
        my = (base[:, 1:2] + n[1] * t[None]).astype(np.float32)
        prof = cv.remap(g, mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE).mean(0)
        lo, hi = float(np.median(prof[: len(t) // 5])), float(np.median(prof[-len(t) // 5:]))
        if hi - lo < 40:
            continue
        p = (prof - lo) / (hi - lo)
        def cross(level):
            j = np.where(p >= level)[0]
            if len(j) == 0 or j[0] == 0:
                return None
            j = j[0]
            return t[j - 1] + (level - p[j - 1]) / (p[j] - p[j - 1] + 1e-9) * (t[j] - t[j - 1])
        x10, x90 = cross(0.1), cross(0.9)
        if x10 is None or x90 is None:
            continue
        sigmas.append(max(float(x90 - x10), 0.25) / 2.563)
    return float(np.median(sigmas)) if sigmas else None
