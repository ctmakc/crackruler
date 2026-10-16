"""Pixel-level evaluation on the held-out TEST split (69 photos) of the Concrete Crack
Segmentation Dataset (Özgenel 2019, CC BY 4.0). Both detectors, coarse 1024-px view.
Metrics: tolerance-2px precision/recall/F1 and strict IoU (micro-averaged over pixels)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from eval.common import seg_pairs, load_split, load_pair, pixel_scores, summarise
from crackruler.segment import segment

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
which = sys.argv[1] if len(sys.argv) > 1 else "test"
split = load_split()
pairs = [p for p in seg_pairs() if p[0] in set(split[which])]
res = {}
for method in ("dnn", "classical"):
    rows, per, times = [], [], []
    for stem, ip, mp in pairs:
        img, gt = load_pair(ip, mp)
        t0 = time.perf_counter()
        m = segment(img, method=method)["mask"]
        times.append(time.perf_counter() - t0)
        sc = pixel_scores(m, gt)
        rows.append(sc)
        s1 = summarise([sc])
        per.append({"image": stem, **s1})
    s = summarise(rows)
    f1s = np.array([p["f1"] for p in per])
    s.update({"per_image_f1_median": round(float(np.median(f1s)), 4),
              "per_image_f1_p10": round(float(np.percentile(f1s, 10)), 4),
              "worst": sorted(per, key=lambda r: r["f1"])[:5],
              "ms_per_image_median": round(1000 * float(np.median(times)), 1)})
    res[method] = s
    print(method, json.dumps({k: v for k, v in s.items() if k != "worst"}), flush=True)
json.dump({"split": which, "n": len(pairs), "tolerance_px": 2, "view_long_side_px": 1024, **res},
          open(os.path.join(OUT, f"seg_{which}.json"), "w"), indent=1)
