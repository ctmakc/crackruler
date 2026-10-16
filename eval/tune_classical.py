"""Grid-search hysteresis thresholds of the classical detector on the TRAIN split."""
import itertools, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2 as cv
from eval.common import seg_pairs, load_split, load_pair, pixel_scores, summarise
from crackruler.segment import ridge_map, hysteresis, filter_components

split = load_split()
pairs = [p for p in seg_pairs() if p[0] in set(split["train"])][:: int(sys.argv[1]) if len(sys.argv) > 1 else 4]
cache = []
for stem, ip, mp in pairs:
    img, gt = load_pair(ip, mp)
    cache.append((ridge_map(cv.cvtColor(img, cv.COLOR_BGR2GRAY)), gt))
import glob, random, numpy as np
from eval.common import DATA
# crack-free patches for the false-positive constraint (disjoint from the eval_cls sample: seed 99, excluded)
eval_neg = set(random.Random(26).sample(sorted(glob.glob(os.path.join(DATA, "cls", "Negative", "*.jpg"))), 1000))
pool = [f for f in sorted(glob.glob(os.path.join(DATA, "cls", "Negative", "*.jpg"))) if f not in eval_neg]
negs = []
for f in random.Random(99).sample(pool, 300):
    im = cv.resize(cv.imread(f), None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    im = cv.copyMakeBorder(im, 8, 8, 8, 8, cv.BORDER_REFLECT)
    negs.append(ridge_map(cv.cvtColor(im, cv.COLOR_BGR2GRAY)))
best = None
for hi, lo in itertools.product([0.4, 0.5, 0.6, 0.7], [0.2, 0.25, 0.3, 0.35]):
    if lo >= hi: continue
    rows = []
    for prob, gt in cache:
        m = filter_components(hysteresis(prob, hi, lo), 40)
        rows.append(pixel_scores(m, gt))
    s = summarise(rows)
    s["neg_fpr"] = round(float(np.mean([filter_components(hysteresis(n, hi, lo), 25)[8:-8, 8:-8].any() for n in negs])), 3)
    print(hi, lo, s, flush=True)
    if s["neg_fpr"] <= 0.05 and (best is None or s["f1"] > best[2]["f1"]):
        best = (hi, lo, s)
print("BEST", json.dumps(best))
