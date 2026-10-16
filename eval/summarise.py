"""Collect committed evaluation results into the figures shown on the page and in the report."""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
seg = json.load(open(os.path.join(R, "seg_test.json")))
cls = json.load(open(os.path.join(R, "cls_sample.json")))
syn = json.load(open(os.path.join(R, "synth_width.json")))["summary"]
resolved = [v for k, v in syn["agent"].items() if v["px_across"] >= 2.4 and v["mae_mm"] is not None]
mae_resolved = max(v["mae_mm"] for v in resolved)
facts = [
    {"value": f"{seg['dnn']['f1']:.3f}", "label": f"pixel F1 (2 px tolerance), U-Net, {seg['n']} held-out real photos"},
    {"value": f"{seg['dnn']['iou']:.3f}", "label": "strict IoU, same test photos"},
    {"value": f"{seg['classical']['f1']:.3f}", "label": "pixel F1 of the classical second detector (no training)"},
    {"value": f"≤{mae_resolved:.3f} mm", "label": "worst mean width error, SYNTHETIC scenes, cracks ≥2.4 px across"},
]
out = os.path.join(os.path.dirname(R), "..", "app", "static", "eval-summary.json")
json.dump(facts, open(out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(facts, indent=1, ensure_ascii=False))
