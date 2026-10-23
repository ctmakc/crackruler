"""Crack segmentation: a classical multi-scale Hessian ridge detector and an ONNX U-Net
run through the OpenCV 5 DNN engine. Both return a probability-like map in [0, 1]."""
from __future__ import annotations

import os
from functools import lru_cache

import cv2 as cv
import numpy as np

MODEL_PATH = os.environ.get("CRACKRULER_MODEL",
                            os.path.join(os.path.dirname(__file__), "..", "models", "crack_unet.onnx"))


# ---------------------------------------------------------------- classical
RIDGE_SCALE = 0.8

def ridge_map(gray: np.ndarray, sigmas=(1.0, 2.0, 3.5), ignore: np.ndarray | None = None) -> np.ndarray:
    """Dark-line (valley) response, Frangi-style, max over scales, robustly normalised."""
    g = gray.astype(np.float32) / 255.0
    # Flatten illumination: divide by a large-scale background estimate.
    bg = cv.GaussianBlur(g, (0, 0), 25)
    g = g / (bg + 1e-3)
    out = np.zeros_like(g)
    for s in sigmas:
        b = cv.GaussianBlur(g, (0, 0), s)
        dxx = cv.Sobel(b, cv.CV_32F, 2, 0, ksize=3) * s * s
        dyy = cv.Sobel(b, cv.CV_32F, 0, 2, ksize=3) * s * s
        dxy = cv.Sobel(b, cv.CV_32F, 1, 1, ksize=3) * s * s
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l1 = 0.5 * (dxx + dyy + tmp)          # largest eigenvalue: across a dark line it is > 0
        l2 = 0.5 * (dxx + dyy - tmp)
        rb = (l2 / (l1 + 1e-6)) ** 2
        v = np.where(l1 > 0, l1 * np.exp(-rb / 0.5), 0).astype(np.float32)
        out = np.maximum(out, v)
    if ignore is not None:
        out[ignore > 0] = 0
    # Absolute scale: the response is a relative-contrast curvature, so a fixed divisor works
    # across images (crack centrelines ~0.45-0.95, background p99.5 ~0.15-0.25 on the train split).
    return np.clip(out / RIDGE_SCALE, 0, 1)


@lru_cache(maxsize=1)
def _net():
    if not os.path.exists(MODEL_PATH):
        return None
    net = cv.dnn.readNetFromONNX(MODEL_PATH)
    return net


def dnn_available() -> bool:
    return _net() is not None


# Tiled inference keeps the DNN engine's working memory bounded (AWS Lambda at 512 MB):
# each tile is TILE px of context, of which the central TILE - 2*MARGIN px are kept.
DNN_TILE = int(os.environ.get("CRACKRULER_DNN_TILE", "512"))     # 0 = whole image in one pass
DNN_MARGIN = int(os.environ.get("CRACKRULER_DNN_MARGIN", "64"))


def dnn_map(bgr: np.ndarray) -> np.ndarray:
    """U-Net crack probability. Input any size. Images larger than one tile are processed in
    overlapping windows; only the centre of each window is kept, so every output pixel away from
    the image edge sees at least DNN_MARGIN px of real context on each side. Windows are clipped
    to the image, so the image edges are handled exactly as in a single pass."""
    h, w = bgr.shape[:2]
    if DNN_TILE <= 0 or (h <= DNN_TILE and w <= DNN_TILE):
        return _dnn_once(bgr)
    core, m = DNN_TILE - 2 * DNN_MARGIN, DNN_MARGIN
    out = np.zeros((h, w), np.float32)
    for y in range(0, h, core):
        for x in range(0, w, core):
            y1, x1 = min(y + core, h), min(x + core, w)
            wy, wx = max(0, y - m), max(0, x - m)
            win = bgr[wy:min(h, y1 + m), wx:min(w, x1 + m)]
            out[y:y1, x:x1] = _dnn_once(win)[y - wy:y1 - wy, x - wx:x1 - wx]
    return out


def _dnn_once(bgr: np.ndarray) -> np.ndarray:
    """One forward pass; the input is padded to a multiple of 16."""
    net = _net()
    if net is None:
        raise RuntimeError("ONNX model not found: " + MODEL_PATH)
    h, w = bgr.shape[:2]
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    img = cv.copyMakeBorder(bgr, 0, ph, 0, pw, cv.BORDER_REFLECT)
    blob = cv.dnn.blobFromImage(img, scalefactor=1 / 255.0, swapRB=True)
    net.setInput(blob)
    logits = net.forward()[0, 0, :h, :w]
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


# ---------------------------------------------------------------- post-processing
def hysteresis(prob: np.ndarray, hi: float, lo: float) -> np.ndarray:
    strong = (prob >= hi).astype(np.uint8)
    weak = (prob >= lo).astype(np.uint8)
    n, lab = cv.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong > 0])] = True
    keep[0] = False
    return (keep[lab] * 255).astype(np.uint8)


def filter_components(mask: np.ndarray, min_len_px: int, min_aspect: float = 4.0) -> np.ndarray:
    """Keep elongated components: centreline length >= min_len_px and
    length / mean width >= min_aspect (mean width = area / centreline length)."""
    n, lab, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    skel = cv.ximgproc.thinning(mask, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
    sk_len = np.bincount(lab[skel > 0], minlength=n).astype(np.float32)
    area = stats[:, cv.CC_STAT_AREA].astype(np.float32)
    extent = np.maximum(stats[:, cv.CC_STAT_WIDTH], stats[:, cv.CC_STAT_HEIGHT])
    keep = (sk_len >= min_len_px) & (extent >= min_len_px) & (sk_len * sk_len >= min_aspect * area)
    keep[0] = False
    return (keep[lab] * 255).astype(np.uint8)


def segment(bgr: np.ndarray, method: str = "dnn", ignore: np.ndarray | None = None,
            hi: float | None = None, lo: float | None = None, min_len_px: int = 40) -> dict:
    if ignore is not None and ignore.any():
        # neutral fill so the marker and the warp border do not look like dark lines
        bgr = bgr.copy()
        fill = np.median(bgr[ignore == 0], axis=0) if (ignore == 0).any() else 128
        bgr[ignore > 0] = fill
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    if method == "dnn":
        prob = dnn_map(bgr)
        hi = 0.5 if hi is None else hi
        lo = 0.3 if lo is None else lo
    elif method == "classical":
        prob = ridge_map(gray, ignore=ignore)
        hi = 0.7 if hi is None else hi     # tuned on the train split, eval/tune_classical.py
        lo = 0.2 if lo is None else lo
    else:
        raise ValueError(method)
    if ignore is not None:
        prob = prob.copy()
        prob[ignore > 0] = 0
    mask = hysteresis(prob, hi, lo)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    mask = filter_components(mask, min_len_px)
    return {"prob": prob, "mask": mask, "method": method, "hi": hi, "lo": lo}
