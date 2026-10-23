"""AWS Lambda entry point (Function URL, payload v2) and shared HTTP routing.

Routes
  GET  /                    single-page inspector UI
  GET  /static/<file>       UI assets, printable marker, sample photos
  GET  /api/health          versions, architecture, model status
  GET  /api/samples         sample photos bundled with the deployment
  POST /api/inspect         run the agent on an uploaded photo or a sample
  GET  /api/report/<id>     fetch a stored report (when REPORT_BUCKET is set)
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
import re
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402

from crackruler import __version__, policy, segment  # noqa: E402
from crackruler.agent import Inspection  # noqa: E402
from crackruler.render import encode_jpeg  # noqa: E402

STATIC = os.path.join(HERE, "static")
SAMPLES = os.path.join(ROOT, "samples")
BUCKET = os.environ.get("REPORT_BUCKET")
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "26000000"))   # 12.6 MP on a 512 MB Lambda
COLD = True
_s3 = None


def _s3c():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client("s3")
    return _s3


def _resp(status, body, ctype="application/json", headers=None, b64=False):
    h = {"content-type": ctype, "cache-control": "no-store", "x-content-type-options": "nosniff"}
    if headers:
        h.update(headers)
    if isinstance(body, (dict, list)):
        body = json.dumps(body)
    return {"statusCode": status, "headers": h, "body": body, "isBase64Encoded": b64}


def _metrics(report, ms, cold):
    """CloudWatch Embedded Metric Format: one log line becomes metrics, no API calls."""
    d = report.get("decision") or {}
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": "CrackRuler", "Dimensions": [["Action"]],
            "Metrics": [{"Name": "LatencyMs", "Unit": "Milliseconds"}, {"Name": "Cracks", "Unit": "Count"},
                        {"Name": "Steps", "Unit": "Count"}, {"Name": "Inspections", "Unit": "Count"}]}]},
        "Action": d.get("action", "none"), "LatencyMs": ms, "Cracks": len(report.get("cracks", [])),
        "Steps": len(report.get("trace", [])), "Inspections": 1, "Calibrated": report.get("calibrated"),
        "ReportId": report.get("id"), "ColdStart": cold, "Arch": platform.machine(),
    }))


def _decode(b: bytes):
    if len(b) > MAX_BYTES:
        raise ValueError("image larger than 8 MB")
    if not (b[:3] == b"\xff\xd8\xff" or b[:8] == b"\x89PNG\r\n\x1a\n"):
        raise ValueError("only JPEG or PNG images are accepted")
    img = cv.imdecode(np.frombuffer(b, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise ValueError(f"image has more than {MAX_PIXELS / 1e6:.1f} megapixels")
    return img


def inspect(payload: dict) -> dict:
    exposure = payload.get("exposure", "humid")
    if exposure not in policy.ACI_224R_TABLE_4_1_MM:
        raise ValueError("unknown exposure class")
    marker_mm = float(payload.get("marker_mm", 50))
    if not 5 <= marker_mm <= 500:
        raise ValueError("marker_mm must be between 5 and 500")
    if payload.get("sample"):
        name = os.path.basename(payload["sample"])
        path = os.path.join(SAMPLES, name)
        if not re.fullmatch(r"[\w.-]+\.(jpg|png)", name) or not os.path.exists(path):
            raise ValueError("unknown sample")
        with open(path, "rb") as f:
            img = _decode(f.read())
    elif payload.get("image_b64"):
        img = _decode(base64.b64decode(payload["image_b64"].split(",")[-1]))
    else:
        raise ValueError("send image_b64 or sample")
    rep = Inspection(exposure=exposure, marker_mm=marker_mm,
                     primary=payload.get("primary", "dnn")).run(img)
    ann = rep.pop("_annotated")
    rep["annotated_jpeg_b64"] = base64.b64encode(encode_jpeg(ann)).decode() if ann is not None else None
    rep["input_size"] = [int(img.shape[1]), int(img.shape[0])]
    if BUCKET:
        key = f"reports/{rep['id']}"
        _s3c().put_object(Bucket=BUCKET, Key=key + ".json", ContentType="application/json",
                          Body=json.dumps({k: v for k, v in rep.items() if k != "annotated_jpeg_b64"}).encode())
        if ann is not None:
            _s3c().put_object(Bucket=BUCKET, Key=key + ".jpg", ContentType="image/jpeg", Body=encode_jpeg(ann))
        rep["stored"] = True
    return rep


def route(method: str, path: str, body: bytes | None):
    global COLD
    cold, COLD = COLD, False
    if method == "GET" and path in ("/", "/index.html"):
        with open(os.path.join(STATIC, "index.html"), "rb") as f:
            return _resp(200, f.read().decode(), "text/html; charset=utf-8",
                         {"content-security-policy": "default-src 'self'; img-src 'self' data: blob:; "
                                                     "style-src 'self' 'unsafe-inline'; script-src 'self'"})
    if method == "GET" and (path.startswith("/static/") or path.startswith("/samples/")):
        base = STATIC if path.startswith("/static/") else SAMPLES
        name = os.path.basename(path)
        p = os.path.join(base, name)
        if not re.fullmatch(r"[\w.-]+", name) or not os.path.isfile(p):
            return _resp(404, {"error": "not found"})
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        with open(p, "rb") as f:
            data = f.read()
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            return _resp(200, data.decode(), ctype, {"cache-control": "public, max-age=3600"})
        return _resp(200, base64.b64encode(data).decode(), ctype, {"cache-control": "public, max-age=3600"}, b64=True)
    if method == "GET" and path == "/api/health":
        return _resp(200, {"ok": True, "version": __version__, "opencv": cv.__version__,
                           "arch": platform.machine(), "python": platform.python_version(),
                           "dnn_model": segment.dnn_available(), "storage": bool(BUCKET),
                           "region": os.environ.get("AWS_REGION"), "cold": cold})
    if method == "GET" and path == "/api/samples":
        items = sorted(n for n in os.listdir(SAMPLES) if n.endswith((".jpg", ".png"))) if os.path.isdir(SAMPLES) else []
        meta = {}
        mp = os.path.join(SAMPLES, "samples.json")
        if os.path.exists(mp):
            meta = json.load(open(mp))
        return _resp(200, [{"name": n, **meta.get(n, {})} for n in items])
    if method == "GET" and path.startswith("/api/report/"):
        rid = path.rsplit("/", 1)[-1]
        if not BUCKET or not re.fullmatch(r"[0-9a-f]{12}", rid):
            return _resp(404, {"error": "not found"})
        try:
            obj = _s3c().get_object(Bucket=BUCKET, Key=f"reports/{rid}.json")
        except Exception:
            return _resp(404, {"error": "not found"})
        return _resp(200, obj["Body"].read().decode())
    if method == "POST" and path == "/api/inspect":
        t0 = time.perf_counter()
        try:
            payload = json.loads(body or b"{}")
            rep = inspect(payload)
        except (ValueError, json.JSONDecodeError) as e:
            print(json.dumps({"level": "warn", "event": "bad_request", "error": str(e)}))
            return _resp(400, {"error": str(e)})
        except Exception as e:  # keep the endpoint up, log the failure
            rid = uuid.uuid4().hex[:8]
            print(json.dumps({"level": "error", "event": "inspect_failed", "ref": rid, "error": repr(e)}))
            return _resp(500, {"error": "internal error", "ref": rid})
        ms = round((time.perf_counter() - t0) * 1000, 1)
        rep["server_ms"] = ms
        rep["cold_start"] = cold
        rep["arch"] = platform.machine()
        _metrics(rep, ms, cold)
        return _resp(200, rep)
    return _resp(404, {"error": "not found"})


def lambda_handler(event, context):
    http = event.get("requestContext", {}).get("http", {})
    method = http.get("method", "GET")
    path = event.get("rawPath", "/")
    body = event.get("body")
    if body is not None:
        body = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode()
    return route(method, path, body)
