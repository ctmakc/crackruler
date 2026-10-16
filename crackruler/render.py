"""Annotated evidence image for the report."""
from __future__ import annotations

import cv2 as cv
import numpy as np

GREEN, AMBER, RED, CYAN = (80, 200, 80), (0, 190, 255), (60, 60, 235), (255, 210, 0)


def annotate(view: np.ndarray, mask: np.ndarray, cracks: list[dict], limit: float | None,
             marker_coarse: np.ndarray | None, unit: str) -> np.ndarray:
    out = view.copy()
    tint = np.zeros_like(out)
    tint[mask > 0] = (255, 0, 255)
    out = cv.addWeighted(out, 1.0, tint, 0.35, 0)
    for c in cracks:
        prof = c.get("_profile", {})
        for (x, y), w in zip(prof.get("pts", []), prof.get("w", [])):
            if w is None:
                col = (160, 160, 160)
            elif limit is None:
                col = CYAN
            elif w > limit:
                col = RED
            elif w > 0.8 * limit:
                col = AMBER
            else:
                col = GREEN
            cv.circle(out, (int(x), int(y)), 1, col, -1, cv.LINE_AA)
        x0, y0, x1, y1 = c["bbox_coarse"]
        cv.rectangle(out, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (255, 255, 255), 1, cv.LINE_AA)
        tag = f"#{c['id']}"
        if c.get("width_p95") is not None:
            tag += f" p95 {c['width_p95']:.2f}{unit}"
        if not c.get("confirmed", True):
            tag += " (unconfirmed)"
        _label(out, tag, (x0 - 4, y0 - 8))
        if c.get("max_at_coarse"):
            cv.drawMarker(out, tuple(c["max_at_coarse"]), (255, 255, 255), cv.MARKER_CROSS, 14, 2, cv.LINE_AA)
    if marker_coarse is not None:
        cv.polylines(out, [marker_coarse.astype(np.int32).reshape(-1, 1, 2)], True, CYAN, 2, cv.LINE_AA)
    return out


def _label(img, text, org, scale=0.62):
    """Readable label: dark plate with light text, kept inside the frame."""
    (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, scale, 1)
    x, y = org
    x = int(min(max(x, 2), img.shape[1] - tw - 10))
    y = int(y if y - th - 8 > 0 else y + th + 30)
    cv.rectangle(img, (x, y - th - 8), (x + tw + 8, y + base), (32, 33, 30), -1)
    cv.putText(img, text, (x + 4, y - 3), cv.FONT_HERSHEY_DUPLEX, scale, (235, 236, 232), 1, cv.LINE_AA)


def encode_jpeg(img: np.ndarray, q: int = 85) -> bytes:
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()
