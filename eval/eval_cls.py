"""Image-level crack / no-crack check on a seeded sample of the Concrete Crack Images for
Classification set (Özgenel 2019, CC BY 4.0; 227x227 patches).

CAVEAT: the patches were cut from the same 458 photos family as the segmentation set, so
for the U-Net this is not an independent test; for the classical detector (no training,
2 thresholds tuned on the train split) it is close to one. Reported for both."""
import glob, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2 as cv
import numpy as np
from eval.common import DATA
from crackruler.segment import segment

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SCALE = 0.5   # patches are native resolution; 0.5 brings crack widths near the training view
rnd = random.Random(26)
pos = rnd.sample(sorted(glob.glob(os.path.join(DATA, "cls", "Positive", "*.jpg"))), N)
neg = rnd.sample(sorted(glob.glob(os.path.join(DATA, "cls", "Negative", "*.jpg"))), N)
res = {}
for method in ("dnn", "classical"):
    tp = fp = fn = tn = 0
    for label, files in ((1, pos), (0, neg)):
        for f in files:
            img = cv.imread(f)
            img = cv.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv.INTER_AREA)
            img = cv.copyMakeBorder(img, 8, 8, 8, 8, cv.BORDER_REFLECT)
            m = segment(img, method=method, min_len_px=25)["mask"][8:-8, 8:-8]
            pred = int(m.any())
            tp += pred and label; fp += pred and not label; fn += (not pred) and label; tn += (not pred) and (not label)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    res[method] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": round((tp + tn) / (2 * N), 4),
                   "precision": round(P, 4), "recall": round(R, 4), "f1": round(2 * P * R / max(P + R, 1e-9), 4)}
    print(method, res[method], flush=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
json.dump({"n_per_class": N, "seed": 26, "scale": SCALE, **res},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cls_sample.json"), "w"), indent=1)
