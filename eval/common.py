"""Shared helpers for the evaluation scripts (dataset loading, split, metrics)."""
from __future__ import annotations

import glob
import json
import os
import random

import cv2 as cv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("CRACKRULER_DATA", os.path.join(ROOT, "data"))
SPLIT = os.path.join(ROOT, "eval", "split.json")
WORK_SIDE = 1024


def seg_pairs():
    pairs = []
    for p in sorted(glob.glob(os.path.join(DATA, "seg", "rgb", "*"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = glob.glob(os.path.join(DATA, "seg", "BW", stem + ".*"))
        if m:
            pairs.append((stem, p, m[0]))
    return pairs


def make_split(seed: int = 26):
    stems = [s for s, _, _ in seg_pairs()]
    rnd = random.Random(seed)
    rnd.shuffle(stems)
    n = len(stems)
    a, b = int(0.7 * n), int(0.85 * n)
    split = {"seed": seed, "train": sorted(stems[:a]), "val": sorted(stems[a:b]), "test": sorted(stems[b:])}
    with open(SPLIT, "w") as f:
        json.dump(split, f, indent=1)
    return split


def load_split():
    if not os.path.exists(SPLIT):
        return make_split()
    with open(SPLIT) as f:
        return json.load(f)


def load_pair(img_path, mask_path, side=WORK_SIDE):
    img = cv.imread(img_path)
    gt = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if gt.shape != img.shape[:2]:
        gt = cv.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)
    s = side / max(img.shape[:2])
    img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    gt = cv.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)
    return img, (gt > 127).astype(np.uint8)


def pixel_scores(pred: np.ndarray, gt: np.ndarray, tol: int = 2):
    """Counts for strict IoU and tolerance-based precision/recall (boundary-tolerant F1)."""
    p = pred > 0
    g = gt > 0
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    g_d = cv.dilate(g.astype(np.uint8), k) > 0
    p_d = cv.dilate(p.astype(np.uint8), k) > 0
    return {
        "tp_p": int((p & g_d).sum()), "np": int(p.sum()),
        "tp_r": int((g & p_d).sum()), "ng": int(g.sum()),
        "inter": int((p & g).sum()), "union": int((p | g).sum()),
    }


def summarise(rows):
    tp_p = sum(r["tp_p"] for r in rows); n_p = sum(r["np"] for r in rows)
    tp_r = sum(r["tp_r"] for r in rows); n_g = sum(r["ng"] for r in rows)
    inter = sum(r["inter"] for r in rows); union = sum(r["union"] for r in rows)
    P = tp_p / max(n_p, 1); R = tp_r / max(n_g, 1)
    F = 2 * P * R / max(P + R, 1e-9)
    return {"precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
            "iou": round(inter / max(union, 1), 4), "images": len(rows)}
