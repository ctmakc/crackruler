import base64
import json
import os
import sys

import cv2 as cv
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from crackruler import calibrate, measure, policy  # noqa: E402
from crackruler.agent import Inspection  # noqa: E402


def _scene(width_px=6.0, marker_px=300, blur=0.8):
    """Fronto-parallel synthetic plate: grey surface, one straight dark band, a marker."""
    img = np.full((1200, 1800), 170, np.float32)
    yy, xx = np.mgrid[0:1200, 0:1800].astype(np.float32)
    d = np.abs(yy - (700 + 0.15 * (xx - 900)))
    cov = np.clip(width_px / 2 + 0.5 - d, 0, 1)
    img = img * (1 - 0.7 * cov * ((xx > 500) & (xx < 1600)))
    m = cv.aruco.generateImageMarker(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50), 0, marker_px)
    img[80 - 40:80 + marker_px + 40, 80 - 40:80 + marker_px + 40] = 235
    img[80:80 + marker_px, 80:80 + marker_px] = np.where(m > 0, 235, 25)
    img = cv.GaussianBlur(img, (0, 0), blur)
    return cv.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv.COLOR_GRAY2BGR)


def test_policy_table_matches_aci_224r():
    assert [v[1] for v in policy.ACI_224R_TABLE_4_1_MM.values()] == [0.41, 0.30, 0.18, 0.15, 0.10]


def test_policy_bands():
    assert policy.decide(0.10, 0.02, "humid", True)["action"] == "log"
    assert policy.decide(0.29, 0.02, "humid", True)["action"] == "monitor"
    assert policy.decide(0.50, 0.02, "humid", True)["action"] == "engineer_review"
    assert policy.decide(0.50, 0.02, "humid", False)["action"] == "human_review"
    assert policy.decide(None, 0.0, "humid", True)["action"] == "no_crack"


def test_marker_scale():
    img = _scene(marker_px=300)
    c, mid = calibrate.detect_marker(img)
    assert mid == 0
    r = calibrate.rectify(img, c, 50.0)
    assert abs(r["px_per_mm"] - 6.0) < 0.05          # 300 px / 50 mm


def test_blur_probe_tracks_sigma():
    for s in (0.8, 1.6):
        img = _scene(blur=s)
        c, _ = calibrate.detect_marker(img)
        r = calibrate.rectify(img, c, 50.0)
        est = calibrate.estimate_psf_sigma(cv.cvtColor(r["image"], cv.COLOR_BGR2GRAY), r["marker_rect"])
        assert abs(est - s) < 0.35, (s, est)


def test_deblur_inverts_model():
    w, sig = np.array([2.0, 4.0, 8.0], np.float32), 1.0
    meas = np.interp(w / sig, measure._R, measure._F) * sig
    corr, ok, _ = measure.deblur_width(meas.astype(np.float32), sig)
    assert ok.all() and np.allclose(corr, w, atol=0.05)


@pytest.mark.parametrize("width_px", [4.0, 9.0])
def test_agent_end_to_end_width(width_px):
    rep = Inspection(exposure="humid", marker_mm=50.0).run(_scene(width_px=width_px))
    assert rep["calibrated"]
    widths = [c["width_median"] for c in rep["cracks"] if c["width_median"]]
    assert widths, rep["trace"]
    true_mm = width_px / 6.0
    assert abs(max(widths) - true_mm) < 0.12, (widths, true_mm)
    assert rep["decision"]["action"] in ("monitor", "engineer_review")
    tools = [t["tool"] for t in rep["trace"]]
    assert tools[:3] == ["assess_quality", "find_marker", "rectify"] and tools[-1] == "decide"


def test_agent_requests_retake_on_blur():
    rep = Inspection().run(_scene(blur=4.0))
    assert rep["decision"]["action"] == "retake"


def test_agent_no_marker_goes_to_human():
    img = _scene()
    img[:500, :500] = 170
    rep = Inspection().run(img)
    assert not rep["calibrated"]
    assert [t["tool"] for t in rep["trace"]].count("find_marker") == 2   # second, enhanced search
    assert rep["decision"]["action"] == "human_review"


def test_handler_routes():
    from app.handler import lambda_handler
    ev = lambda m, p, b=None: {"rawPath": p, "requestContext": {"http": {"method": m}}, "body": b, "isBase64Encoded": False}
    h = json.loads(lambda_handler(ev("GET", "/api/health"), None)["body"])
    assert h["ok"] and h["opencv"].startswith("5.") and h["dnn_model"]
    assert lambda_handler(ev("GET", "/"), None)["statusCode"] == 200
    assert lambda_handler(ev("GET", "/static/../handler.py"), None)["statusCode"] == 404
    bad = lambda_handler(ev("POST", "/api/inspect", json.dumps({"image_b64": base64.b64encode(b"GIF89a").decode()})), None)
    assert bad["statusCode"] == 400
    ok, buf = cv.imencode(".jpg", _scene())
    r = lambda_handler(ev("POST", "/api/inspect", json.dumps({"image_b64": base64.b64encode(buf.tobytes()).decode()})), None)
    body = json.loads(r["body"])
    assert r["statusCode"] == 200 and body["calibrated"] and body["annotated_jpeg_b64"]
