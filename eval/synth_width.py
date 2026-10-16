"""SYNTHETIC metric-accuracy test (see eval/synth.py: real crack-free concrete backgrounds
from TEST-split photos, synthetic crack of known constant width, synthetic ArUco marker,
random camera tilt / blur / noise / JPEG).

Two numbers per scene:
  oracle   - width measured along the TRUE centreline (mapped through the camera and the
             agent's own rectification); isolates the scale + profile-width method.
  agent    - full end-to-end agent run (detection, cross-check, measurement); p50 width of
             the longest confirmed crack.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2 as cv
import numpy as np
from eval.synth import render_scene
from crackruler.agent import Inspection
from crackruler import measure

WIDTHS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
PPMS = [5, 8, 12]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
rng = np.random.default_rng(2026)
rows = []
for w in WIDTHS:
    for ppm in PPMS:
        for r in range(REPS):
            tilt = float(rng.uniform(0, 30)); blur = float(rng.uniform(0.6, 1.2))
            sc = render_scene(rng, w, ppm, tilt_deg=tilt, blur=blur, noise=float(rng.uniform(1.5, 4)))
            ins = Inspection(exposure="humid")
            rep = ins.run(sc["image"])
            row = {"width_mm": w, "cam_ppm": ppm, "tilt": round(tilt, 1), "blur": round(blur, 2),
                   "background": sc["background"], "marker_found": rep["calibrated"],
                   "est_ppm": rep["px_per_mm"], "action": rep["decision"]["action"]}
            if rep["calibrated"]:
                H2 = ins.s["rect"]["H"]
                pts_w = sc["path_world_px"][5:-5:4].reshape(-1, 1, 2)
                pts_cam = cv.perspectiveTransform(pts_w, sc["H_world_to_cam"].astype(np.float64))
                pts_r = cv.perspectiveTransform(pts_cam, H2).reshape(-1, 2).astype(np.float32)
                d = np.gradient(pts_r, axis=0); d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
                gray = cv.cvtColor(ins.s["work"], cv.COLOR_BGR2GRAY)
                ppm_r = ins.s["px_per_mm"]
                half = np.full(len(pts_r), max(8.0, 3 * w * ppm_r + 6), np.float32)
                wp, _ = measure.profile_widths(gray, pts_r, d.astype(np.float32), half)
                ok = ~np.isnan(wp)
                row["psf_sigma"] = round(ins.s["psf_sigma"], 3) if ins.s.get("psf_sigma") else None
                if ok.any():
                    wp[ok], _, _ = measure.deblur_width(wp[ok], ins.s.get("psf_sigma"))
                row["oracle_valid"] = round(float(ok.mean()), 3)
                row["oracle_p50_mm"] = round(float(np.median(wp[ok])) / ppm_r, 4) if ok.any() else None
                row["px_across"] = round(w * ppm_r, 2)
                conf = [c for c in rep["cracks"] if c.get("confirmed") and c.get("width_median") is not None]
                sc_hr = ins.s["coarse"]["scale_hr"]
                pc = pts_r / sc_hr

                def hits(c):
                    x0, y0, x1, y1 = c["bbox_coarse"]
                    return int(((pc[:, 0] >= x0 - 3) & (pc[:, 0] <= x1 + 3) & (pc[:, 1] >= y0 - 3) & (pc[:, 1] <= y1 + 3)).sum())
                conf = [c for c in conf if hits(c) > 0]
                row["other_confirmed"] = len([c for c in rep["cracks"] if c.get("confirmed")]) - len(conf)
                if conf:
                    best = max(conf, key=hits)
                    row["agent_p50_mm"] = best["width_median"]
                    row["agent_len_mm"] = best["length"]
                else:
                    row["agent_p50_mm"] = None
            rows.append(row)
            print(json.dumps(row), flush=True)


def stats(key):
    out = {}
    for w in WIDTHS:
        for ppm in PPMS:
            sel = [r for r in rows if r["width_mm"] == w and r["cam_ppm"] == ppm]
            vals = [r[key] for r in sel if r.get(key) is not None]
            e = np.array(vals) - w
            out[f"{w}mm@{ppm}ppm"] = {"n": len(sel), "measured": len(vals),
                                      "px_across": round(w * ppm, 2),
                                      "bias_mm": round(float(e.mean()), 3) if len(e) else None,
                                      "mae_mm": round(float(np.abs(e).mean()), 3) if len(e) else None,
                                      "rel_mae": round(float(np.abs(e).mean() / w), 3) if len(e) else None}
    return out


summary = {"SYNTHETIC": True, "reps": REPS, "marker_found_rate": round(np.mean([r["marker_found"] for r in rows]), 3),
           "oracle": stats("oracle_p50_mm"), "agent": stats("agent_p50_mm")}
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "synth_width.json"), "w"), indent=1)
print(json.dumps(summary["oracle"])); print(json.dumps(summary["agent"]))
