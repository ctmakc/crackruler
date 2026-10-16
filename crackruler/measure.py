"""Crack geometry: skeleton, length, orientation and sub-pixel width from intensity profiles.

Width is measured as the full width at half depth (FWHD) of the grey-level profile taken
perpendicular to the crack centreline, sampled at 0.25 px with cv.remap on the
highest-resolution view available. Profiles narrower than ~2 px are limited by optics
(blur, demosaicing) and are reported as "at or below resolution".
"""
from __future__ import annotations

import math

import cv2 as cv
import numpy as np

STEP = 0.25          # profile sampling step (px)
MIN_DEPTH = 8.0      # grey levels; shallower profiles are not trusted
RES_LIMIT_PX = 2.0   # widths below this many pixels are optics-limited


# FWHD of a box profile of width w blurred by a Gaussian sigma, measured at half of the
# *observed* depth: FWHD / sigma = F(w / sigma). Tabulated once, inverted by interpolation.
def _fwhd_table():
    from math import erf, sqrt
    r = np.concatenate([np.linspace(0.01, 4, 161), np.linspace(4.05, 40, 720)])
    F = []
    for w in r:
        x = np.linspace(0, w / 2 + 6, 4000)
        p = 0.5 * (np.vectorize(erf)((x + w / 2) / sqrt(2)) - np.vectorize(erf)((x - w / 2) / sqrt(2)))
        half = p[0] / 2
        j = int(np.argmax(p < half))
        xj = x[j - 1] + (p[j - 1] - half) / (p[j - 1] - p[j] + 1e-12) * (x[j] - x[j - 1])
        F.append(2 * xj)
    return r, np.array(F)


_R, _F = _fwhd_table()
UNRESOLVED_RATIO = 1.10      # FWHD within 10 % of a point-source blur -> width not resolvable


def deblur_width(w_px: np.ndarray, sigma: float | None):
    """Invert the blur model. Returns (corrected width px, resolved flag, upper bound px)."""
    if sigma is None or sigma <= 0:
        return w_px, np.ones_like(w_px, bool), None
    m = w_px / sigma
    f0 = _F[0] * UNRESOLVED_RATIO
    ub = float(np.interp(f0, _F, _R) * sigma)
    corr = np.interp(m, _F, _R) * sigma
    resolved = m > f0
    corr = np.where(resolved, corr, ub)
    return corr.astype(np.float32), resolved, ub


def skeletonize(mask: np.ndarray) -> np.ndarray:
    return cv.ximgproc.thinning(mask, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)


def _tangents(skel_pts: np.ndarray, radius: int = 6) -> np.ndarray:
    """Local tangent per skeleton point by PCA over neighbours (vectorised via a grid lookup)."""
    tang = np.zeros((len(skel_pts), 2), np.float32)
    if len(skel_pts) == 0:
        return tang
    from collections import defaultdict
    cell = defaultdict(list)
    for i, (x, y) in enumerate(skel_pts):
        cell[(x // radius, y // radius)].append(i)
    for i, (x, y) in enumerate(skel_pts):
        cx, cy = x // radius, y // radius
        idx = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in cell.get((cx + dx, cy + dy), ())]
        nb = skel_pts[idx].astype(np.float32)
        nb = nb[np.hypot(nb[:, 0] - x, nb[:, 1] - y) <= radius]
        if len(nb) < 3:
            tang[i] = (1, 0)
            continue
        c = nb - nb.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        tang[i] = vt[0]
    return tang


def profile_widths(gray: np.ndarray, pts: np.ndarray, tang: np.ndarray, half_len: np.ndarray):
    """FWHD widths (px) at each point; NaN where the profile is unusable."""
    n = len(pts)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    L = float(np.max(half_len))
    t = np.arange(-L, L + STEP / 2, STEP, dtype=np.float32)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    mx = (pts[:, 0:1] + nrm[:, 0:1] * t[None]).astype(np.float32)
    my = (pts[:, 1:2] + nrm[:, 1:2] * t[None]).astype(np.float32)
    prof = cv.remap(gray.astype(np.float32), mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    widths = np.full(n, np.nan, np.float32)
    depth = np.zeros(n, np.float32)
    c0 = len(t) // 2
    for i in range(n):
        k = int(half_len[i] / STEP)
        p = prof[i, c0 - k:c0 + k + 1]
        q = max(2, len(p) // 6)
        bg = float(np.median(np.concatenate([p[:q], p[-q:]])))
        mid = p[len(p) // 3: 2 * len(p) // 3]
        j = int(np.argmin(mid)) + len(p) // 3
        core = float(p[j])
        d = bg - core
        depth[i] = d
        if d < MIN_DEPTH:
            continue
        half = bg - d / 2
        l = j
        while l > 0 and p[l] < half:
            l -= 1
        r = j
        while r < len(p) - 1 and p[r] < half:
            r += 1
        if l == 0 or r == len(p) - 1:
            continue
        # linear interpolation of both crossings
        xl = l + (half - p[l]) / (p[l + 1] - p[l] + 1e-6)
        xr = r - (half - p[r]) / (p[r - 1] - p[r] + 1e-6)
        widths[i] = max(xr - xl, 0) * STEP
    return widths, depth


def measure(mask: np.ndarray, gray_hr: np.ndarray, scale_hr: float, px_per_mm: float | None,
            min_len_mm: float = 10.0, sample_every: int = 3, psf_sigma: float | None = None) -> list[dict]:
    """Measure every crack component.

    mask      : binary mask on the coarse view.
    gray_hr   : the high-resolution grey image (scale_hr = hr_px / coarse_px).
    px_per_mm : pixel density of the high-resolution view, None if uncalibrated.
    """
    skel = skeletonize(mask)
    dt = cv.distanceTransform(mask, cv.DIST_L2, 5)
    n, lab = cv.connectedComponents(skel, connectivity=8)
    cracks = []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 5:
            continue
        pts_c = np.stack([xs, ys], 1)
        tang = _tangents(pts_c)
        # length: count 8-neighbour steps on the skeleton
        sub = (lab == i).astype(np.uint8)
        diag = cv.filter2D(sub, -1, np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.float32))[ys, xs].sum() / 2
        orth = cv.filter2D(sub, -1, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32))[ys, xs].sum() / 2
        length_px_c = float(orth + diag * math.sqrt(2))
        length_hr = length_px_c * scale_hr
        if px_per_mm and length_hr / px_per_mm < min_len_mm:
            continue
        sel = np.arange(0, len(pts_c), sample_every)
        pts_hr = (pts_c[sel].astype(np.float32) + 0.5) * scale_hr - 0.5
        half_len = np.clip((dt[ys[sel], xs[sel]] * 3 + 4) * scale_hr, 6, 80).astype(np.float32)
        w_raw, depth = profile_widths(gray_hr, pts_hr, tang[sel], half_len)
        ok = ~np.isnan(w_raw)
        w_px = w_raw.copy()
        resolved = np.ones(len(w_px), bool)
        if ok.any():
            w_px[ok], resolved[ok], _ = deblur_width(w_raw[ok], psf_sigma)
        if ok.sum() < 3:
            w_valid = np.array([], np.float32)
        else:
            w_valid = w_px[ok]
        _, _, vt = np.linalg.svd(pts_c - pts_c.mean(0), full_matrices=False)
        ang = (math.degrees(math.atan2(vt[0][1], vt[0][0])) + 180) % 180
        unit = (1.0 / px_per_mm) if px_per_mm else 1.0
        rec = {
            "id": len(cracks) + 1,
            "length": round(length_hr * unit, 1),
            "orientation_deg": round(ang, 1),
            "samples": int(len(sel)), "valid_samples": int(ok.sum()),
            "bbox_coarse": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "max_at_coarse": None,
        }
        if len(w_valid):
            p95 = float(np.percentile(w_valid, 95))
            rec.update({
                "width_median": round(float(np.median(w_valid)) * unit, 3),
                "width_p95": round(p95 * unit, 3),
                "width_max": round(float(w_valid.max()) * unit, 3),
                "below_resolution_fraction": round(float(np.mean(~resolved[ok])) if psf_sigma else
                                                   float(np.mean(w_valid < RES_LIMIT_PX)), 3),
                "width_raw_p95": round(float(np.percentile(w_raw[ok], 95)) * unit, 3),
            })
            j = int(np.nanargmax(np.where(ok, w_px, -1)))
            rec["max_at_coarse"] = [int(pts_c[sel][j][0]), int(pts_c[sel][j][1])]
        else:
            rec.update({"width_median": None, "width_p95": None, "width_max": None,
                        "below_resolution_fraction": None})
        rec["_profile"] = {"pts": pts_c[sel].tolist(), "w": [None if np.isnan(v) else float(v * unit) for v in w_px]}
        cracks.append(rec)
    return cracks
