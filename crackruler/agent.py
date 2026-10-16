"""CrackRuler inspection agent: a perception -> decision -> action loop over OpenCV 5 tools.

The planner is a small, deterministic, auditable policy. Every step records what the
tool observed and why the next tool was chosen, so an inspector can replay the reasoning.
Visual evidence changes the plan: a blurry frame ends in a retake request, a missing
marker triggers an enhanced second search and then pixel-only mode, an empty first pass
triggers a second detector, disagreement between detectors marks cracks as unconfirmed,
and insufficient pixel density turns a "log" verdict into a request for a closer photo.
"""
from __future__ import annotations

import math
import time
import uuid

import cv2 as cv
import numpy as np

from . import calibrate, measure, policy, quality, render, segment

COARSE_SIDE = 1024
PSF_MAX_PX = 2.5         # marker-edge blur above this: defocus or motion blur, retake
SIGMA_PX = 0.45          # 1-sigma width error in px from eval/synth_width.py (oracle centreline)


class Inspection:
    def __init__(self, exposure: str = "humid", marker_mm: float = 50.0, primary: str = "dnn",
                 max_steps: int = 14):
        if exposure not in policy.ACI_224R_TABLE_4_1_MM:
            raise ValueError(f"unknown exposure {exposure}")
        self.exposure, self.marker_mm, self.max_steps = exposure, float(marker_mm), max_steps
        self.primary = primary if (primary != "dnn" or segment.dnn_available()) else "classical"
        self.secondary = "classical" if self.primary == "dnn" else ("dnn" if segment.dnn_available() else None)
        self.trace: list[dict] = []
        self.s: dict = {}

    # ------------------------------------------------------------------ tools
    def t_assess_quality(self):
        q = quality.assess(self.s["input"])
        self.s["quality"] = q
        return q, ("frame usable" if q["ok"] else "frame has issues: " + ", ".join(q["issues"]))

    def t_find_marker(self, enhanced: bool = False):
        img = self.s["input"]
        if enhanced:
            g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            g = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
            corners, mid = calibrate.detect_marker(g)
        else:
            corners, mid = calibrate.detect_marker(img)
        self.s["marker_attempts"] = self.s.get("marker_attempts", 0) + 1
        if corners is not None:
            self.s["marker"] = {"corners": corners, "id": mid}
            side = float(np.linalg.norm(corners - np.roll(corners, -1, 0), axis=1).mean())
            return {"found": True, "id": mid, "side_px": round(side, 1)}, "marker found, metric scale available"
        return {"found": False, "enhanced": enhanced}, "no marker in view"

    def t_rectify(self):
        r = calibrate.rectify(self.s["input"], self.s["marker"]["corners"], self.marker_mm)
        self.s["rect"] = r
        self.s["work"] = r["image"]
        self.s["px_per_mm"] = r["px_per_mm"]
        # ignore the marker and its paper margin (up to half a marker side around it)
        self.s["ignore_hr"] = calibrate.marker_mask(r["image"].shape, r["marker_rect"], pad_px=int(0.5 * r["marker_px"]) + 4)
        self.s["ignore_hr"][r["valid"] == 0] = 255
        sig = calibrate.estimate_psf_sigma(cv.cvtColor(r["image"], cv.COLOR_BGR2GRAY), r["marker_rect"])
        self.s["psf_sigma"] = sig
        obs = {"px_per_mm": round(r["px_per_mm"], 3), "rel_scale_unc": round(r["rel_scale_unc"], 4),
               "psf_sigma_px": None if sig is None else round(sig, 3),
               "marker_skew": round(r["marker_skew"], 4), "size": list(r["image"].shape[1::-1])}
        return obs, f"surface rectified at {r['px_per_mm']:.2f} px/mm"

    def _coarse(self):
        if "coarse" not in self.s:
            work = self.s["work"]
            k = COARSE_SIDE / max(work.shape[:2])
            k = min(k, 1.0)
            view = cv.resize(work, None, fx=k, fy=k, interpolation=cv.INTER_AREA) if k < 1 else work.copy()
            ign = np.zeros(view.shape[:2], np.uint8)
            if self.s.get("ignore_hr") is not None:
                ign = cv.resize(self.s["ignore_hr"], (view.shape[1], view.shape[0]), interpolation=cv.INTER_NEAREST)
                ign = cv.dilate(ign, np.ones((15, 15), np.uint8))
            b = 8   # frame edge: padding artefacts, not evidence
            ign[:b, :] = ign[-b:, :] = 255
            ign[:, :b] = ign[:, -b:] = 255
            self.s["coarse"] = {"view": view, "scale_hr": 1.0 / k, "ignore": ign}
        return self.s["coarse"]

    def t_segment(self, method: str, sensitive: bool = False):
        c = self._coarse()
        kw = {}
        if sensitive:
            kw = {"hi": 0.35, "lo": 0.15} if method == "dnn" else {"hi": 0.5, "lo": 0.15}
        seg = segment.segment(c["view"], method=method, ignore=c["ignore"], **kw)
        key = "seg_" + method + ("_sensitive" if sensitive else "")
        self.s[key] = seg
        n, _ = cv.connectedComponents(seg["mask"])
        frac = float((seg["mask"] > 0).mean())
        return {"method": method, "components": n - 1, "mask_fraction": round(frac, 5)}, \
            f"{method} found {n - 1} candidate crack region(s)"

    def t_cross_check(self):
        prim = self.s["seg_" + self.primary]["mask"]
        sec = self.s["seg_" + self.secondary + "_sensitive"]["mask"] if self.secondary else None
        n, lab = cv.connectedComponents(prim)
        support = {}
        if sec is not None:
            sec_d = cv.dilate(sec, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
            sk = measure.skeletonize(prim)
            for i in range(1, n):
                m = (lab == i) & (sk > 0)
                support[i] = float((sec_d[m] > 0).mean()) if m.any() else 0.0
        self.s["support_lab"] = lab
        self.s["support"] = support
        weak = [i for i, v in support.items() if v < 0.3]
        return {"support": {str(k): round(v, 2) for k, v in support.items()}, "weak": len(weak)}, \
            ("detectors agree" if not weak else f"{len(weak)} region(s) not confirmed by the second detector")

    def t_measure(self):
        c = self._coarse()
        gray_hr = cv.cvtColor(self.s["work"], cv.COLOR_BGR2GRAY)
        cracks = measure.measure(self.s["seg_" + self.primary]["mask"], gray_hr, c["scale_hr"],
                                 self.s.get("px_per_mm"),
                                 min_len_mm=10.0 if self.s.get("px_per_mm") else 0.0,
                                 psf_sigma=self.s.get("psf_sigma"))
        lab, sup = self.s.get("support_lab"), self.s.get("support", {})
        for cr in cracks:
            if lab is not None and sup:
                x0, y0, x1, y1 = cr["bbox_coarse"]
                ids = np.unique(lab[y0:y1 + 1, x0:x1 + 1])
                vals = [sup[i] for i in ids if i in sup]
                cr["support"] = round(max(vals), 2) if vals else 0.0
                cr["confirmed"] = cr["support"] >= 0.3
            else:
                cr["support"], cr["confirmed"] = None, True
        self.s["cracks"] = cracks
        unit = "mm" if self.s.get("px_per_mm") else "px"
        widest = max([c["width_p95"] for c in cracks if c["width_p95"] is not None], default=None)
        return {"cracks": len(cracks), "widest_p95": widest, "unit": unit}, \
            f"measured {len(cracks)} crack(s) on the full-resolution view" + (f", widest p95 {widest:.2f} {unit}" if widest else "")

    def t_check_resolution(self):
        ppm = self.s["px_per_mm"]
        sig = self.s.get("psf_sigma")
        if sig:
            _, _, ub = measure.deblur_width(np.array([0.0], np.float32), sig)
            min_res = ub / ppm
        else:
            min_res = measure.RES_LIMIT_PX / ppm
        limit = policy.guideline_mm(self.exposure)
        ok = min_res <= limit
        self.s["resolution"] = {"min_resolvable_mm": round(min_res, 3), "limit_mm": limit, "can_resolve_limit": ok,
                                "closer_factor": None if ok else round(min_res / limit * 1.15, 1)}
        return self.s["resolution"], ("pixel density resolves the guideline width" if ok else
                                      f"one pixel pair spans {min_res:.2f} mm, wider than the {limit} mm guideline")

    def t_decide(self):
        cal = self.s.get("px_per_mm") is not None
        cracks = self.s.get("cracks", [])
        conf = [c for c in cracks if c.get("confirmed", True) and c["width_p95"] is not None]
        unconf = [c for c in cracks if not c.get("confirmed", True)]
        worst = max(conf, key=lambda c: c["width_p95"], default=None)
        w = worst["width_p95"] if worst else None
        unc = 0.0
        if cal and w is not None:
            rel = self.s["rect"]["rel_scale_unc"]
            unc = math.sqrt((rel * w) ** 2 + (SIGMA_PX / self.s["px_per_mm"]) ** 2) * 2  # ~95 %
        d = policy.decide(w, unc, self.exposure, cal)
        res = self.s.get("resolution")
        thin = worst is not None and res is not None and worst["width_median"] < 2 * res["min_resolvable_mm"]
        if res and not res["can_resolve_limit"] and (d["action"] == "log" or thin):
            d = {**d, "action": "retake",
                 "reason": f"The photo cannot resolve the {res['limit_mm']} mm guideline (smallest measurable width "
                           f"{res['min_resolvable_mm']:.2f} mm); move about {res['closer_factor']}x closer. "
                           f"Widths measured on this frame are upper bounds only."}
        if d["action"] in ("no_crack", "log") and unconf:
            d = {**d, "action": "human_review",
                 "reason": f"{len(unconf)} region(s) seen by only one detector need a human look"}
        d["width_p95_mm" if cal else "width_p95_px"] = w
        d["uncertainty_mm"] = round(unc, 3) if cal else None
        d["worst_crack"] = worst["id"] if worst else None
        d["needs_human"] = d["action"] in ("engineer_review", "human_review", "retake")
        d["next_step"] = policy.ACTIONS[d["action"]]
        self.s["decision"] = d
        return d, f"action: {d['action']}"

    # ------------------------------------------------------------------ planner
    def plan(self) -> tuple[str, dict]:
        s = self.s
        if "quality" not in s:
            return "assess_quality", {}
        if not s["quality"]["ok"] and "decision" not in s:
            s["retake_reason"] = "image quality gate failed: " + ", ".join(s["quality"]["issues"])
            return "request_retake", {}
        if "marker" not in s and s.get("marker_attempts", 0) == 0:
            return "find_marker", {}
        if "marker" not in s and s.get("marker_attempts", 0) == 1:
            return "find_marker", {"enhanced": True}
        if "marker" in s and "rect" not in s:
            return "rectify", {}
        if s.get("psf_sigma") and s["psf_sigma"] > PSF_MAX_PX and "decision" not in s:
            s["retake_reason"] = (f"marker edge shows blur sigma {s['psf_sigma']:.1f} px (limit {PSF_MAX_PX}); "
                                  "hold the phone steady and refocus")
            return "request_retake", {}
        if "work" not in s:
            s["work"], s["ignore_hr"] = s["input"], None
        if "seg_" + self.primary not in s:
            return "segment", {"method": self.primary}
        empty = not s["seg_" + self.primary]["mask"].any()
        if self.secondary and "seg_" + self.secondary + "_sensitive" not in s:
            return "segment", {"method": self.secondary, "sensitive": True}
        if empty:
            sec = s.get("seg_" + (self.secondary or "") + "_sensitive")
            if sec is not None and sec["mask"].any() and "decision" not in s:
                return "flag_disagreement", {}
            if "decision" not in s:
                return "decide", {}
            return "finish", {}
        if "support" not in s:
            return "cross_check", {}
        if "cracks" not in s:
            return "measure", {}
        if s.get("px_per_mm") and "resolution" not in s:
            return "check_resolution", {}
        if "decision" not in s:
            return "decide", {}
        return "finish", {}

    def run(self, bgr: np.ndarray) -> dict:
        t_all = time.perf_counter()
        self.s = {"input": bgr}
        self.trace = []
        tools = {
            "assess_quality": self.t_assess_quality, "find_marker": self.t_find_marker,
            "rectify": self.t_rectify, "segment": self.t_segment, "cross_check": self.t_cross_check,
            "measure": self.t_measure, "check_resolution": self.t_check_resolution, "decide": self.t_decide,
        }
        for step in range(1, self.max_steps + 1):
            name, args = self.plan()
            t0 = time.perf_counter()
            if name == "finish":
                break
            if name == "request_retake":
                self.s["decision"] = {"action": "retake", "needs_human": True, "limit_mm": policy.guideline_mm(self.exposure),
                                      "reason": self.s["retake_reason"], "next_step": policy.ACTIONS["retake"]}
                obs, why = {"reason": self.s["retake_reason"]}, "stop: measuring widths on this frame would be guesswork"
            elif name == "flag_disagreement":
                self.s["decision"] = {"action": "human_review", "needs_human": True,
                                      "limit_mm": policy.guideline_mm(self.exposure),
                                      "reason": "primary detector found nothing, sensitive second detector found candidates",
                                      "next_step": policy.ACTIONS["human_review"]}
                obs, why = {}, "detectors disagree on an empty result"
            else:
                obs, why = tools[name](**args)
            self.trace.append({"step": step, "tool": name, "args": args, "observation": _jsonable(obs),
                               "note": why, "ms": round((time.perf_counter() - t0) * 1000, 1)})
        else:
            self.trace.append({"step": self.max_steps + 1, "tool": "abort", "note": "step budget exhausted"})
            self.s.setdefault("decision", {"action": "human_review", "needs_human": True, "reason": "step budget exhausted",
                                           "next_step": policy.ACTIONS["human_review"]})
        return self._report(time.perf_counter() - t_all)

    def _report(self, elapsed: float) -> dict:
        s = self.s
        cal = s.get("px_per_mm") is not None
        unit = "mm" if cal else "px"
        cracks = s.get("cracks", [])
        annotated = None
        if "coarse" in s:
            c = s["coarse"]
            mk = None
            if s.get("rect") is not None:
                mk = s["rect"]["marker_rect"] / c["scale_hr"]
            mask = s.get("seg_" + self.primary, {}).get("mask", np.zeros(c["view"].shape[:2], np.uint8))
            annotated = render.annotate(c["view"], mask, cracks,
                                        policy.guideline_mm(self.exposure) if cal else None, mk, unit)
        elif "input" in s:
            annotated = s["input"]
        public = [{k: v for k, v in cr.items() if not k.startswith("_")} for cr in cracks]
        return {
            "id": uuid.uuid4().hex[:12],
            "engine": {"opencv": cv.__version__, "primary": self.primary, "secondary": self.secondary},
            "exposure": {"key": self.exposure, "label": policy.ACI_224R_TABLE_4_1_MM[self.exposure][0],
                         "guideline_mm": policy.guideline_mm(self.exposure),
                         "source": "ACI 224R-01 (Reapproved 2008), Table 4.1"},
            "calibrated": cal, "unit": unit,
            "px_per_mm": round(s["px_per_mm"], 3) if cal else None,
            "quality": s.get("quality"), "resolution": s.get("resolution"),
            "cracks": public, "decision": _jsonable(s.get("decision")), "trace": self.trace,
            "elapsed_ms": round(elapsed * 1000, 1),
            "_annotated": annotated,
        }


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
