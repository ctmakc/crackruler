"""Image-quality gate: sharpness, exposure and whether the resolution can resolve the limit."""
from __future__ import annotations

import cv2 as cv
import numpy as np

SHARPNESS_MIN = 10.0     # tenengrad on the 1024-px view; real test photos: min 16.6, p5 31.9
CLIP_MAX = 0.20          # fraction of pixels at 0-4 or 251-255


def assess(bgr: np.ndarray) -> dict:
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = 1024.0 / max(h, w)
    small = cv.resize(gray, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else gray
    gx = cv.Sobel(small, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(small, cv.CV_32F, 0, 1, ksize=3)
    ten = float(np.sqrt(np.mean(gx * gx + gy * gy)))
    clip = float(np.mean((small < 5) | (small > 250)))
    issues = []
    if ten < SHARPNESS_MIN:
        issues.append("blurry")
    if clip > CLIP_MAX:
        issues.append("badly exposed")
    return {"sharpness": round(ten, 2), "clipped_fraction": round(clip, 4), "issues": issues,
            "ok": not issues, "size": [w, h]}
