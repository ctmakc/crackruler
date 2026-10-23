"""Latency per agent tool. Local: runs the agent in-process on the bundled samples.
Remote: `python eval/bench.py https://<function-url>` posts each sample N times."""
import json, os, statistics, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = int(os.environ.get("N", 5))
SAMPLES = ["synthetic-0.35mm.jpg", "synthetic-0.60mm-tilted.jpg", "real-191.jpg"]
per_tool, total, rtt = {}, [], []
if len(sys.argv) > 1:
    url = sys.argv[1].rstrip("/") + "/api/inspect"
    for s in SAMPLES:
        for _ in range(N):
            t0 = time.perf_counter()
            req = urllib.request.Request(url, data=json.dumps({"sample": s}).encode(), headers={"content-type": "application/json"})
            rep = json.load(urllib.request.urlopen(req, timeout=90))
            rtt.append((time.perf_counter() - t0) * 1000)
            total.append(rep["server_ms"])
            for t in rep["trace"]:
                per_tool.setdefault(t["tool"] + ("·" + t["args"]["method"] if t.get("args", {}).get("method") else ""), []).append(t.get("ms", 0))
    where = {"endpoint": sys.argv[1], "arch": rep.get("arch")}
else:
    import cv2 as cv
    from crackruler.agent import Inspection
    for s in SAMPLES:
        img = cv.imread(os.path.join("samples", s))
        for _ in range(N):
            rep = Inspection().run(img)
            total.append(rep["elapsed_ms"])
            for t in rep["trace"]:
                per_tool.setdefault(t["tool"] + ("·" + t["args"]["method"] if t.get("args", {}).get("method") else ""), []).append(t.get("ms", 0))
    import platform
    where = {"local": platform.machine(), "threads": cv.getNumThreads()}
out = {**where, "n": len(total), "server_ms_median": round(statistics.median(total), 1),
       "rtt_ms_median": round(statistics.median(rtt), 1) if rtt else None,
       "per_tool_ms_median": {k: round(statistics.median(v), 1) for k, v in per_tool.items()}}
print(json.dumps(out, indent=1))
name = os.environ.get("OUT")
if name:
    os.makedirs("eval/results", exist_ok=True)
    json.dump(out, open(os.path.join("eval/results", name), "w"), indent=1)
