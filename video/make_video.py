"""Build the submission video: edge-tts narration + live recordings of the deployed UI + slides,
burned-in subtitles and an SRT for YouTube. Usage:
  BASE=https://<function-url> python video/make_video.py        (BASE defaults to a local server)
Outputs in $VIDEO_OUT (default /data/cache/crackruler/video-out): crackruler_demo.mp4, .srt, frames/."""
import asyncio, json, os, re, subprocess, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenes import SCENES

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BASE", "http://127.0.0.1:18731").rstrip("/")
OUT = os.environ.get("VIDEO_OUT", "/data/cache/crackruler/video-out")
TTS = os.environ.get("EDGE_TTS", "/data/venvs/edge-tts/bin/edge-tts")
VOICE = os.environ.get("VOICE", "en-US-AndrewNeural")
NODE_PATH = os.environ.get("NODE_PATH", "/home/llm/.local/lib/node_modules")
LEAD, TAIL = 0.35, 0.55
os.makedirs(OUT, exist_ok=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def dur(p):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]).strip())


def check_expectations():
    for s in SCENES:
        if "expect" not in s:
            continue
        req = urllib.request.Request(BASE + "/api/inspect", data=json.dumps({"sample": s["sample"]}).encode(),
                                     headers={"content-type": "application/json"})
        rep = json.load(urllib.request.urlopen(req, timeout=120))
        act = rep["decision"]["action"]
        assert act == s["expect"]["action"], (s["id"], act)
        if "median" in s["expect"]:
            meds = [c["width_median"] for c in rep["cracks"] if c.get("confirmed") and c["width_median"]]
            assert meds and abs(round(max(meds), 2) - s["expect"]["median"]) <= 0.011, (s["id"], meds)
        print("ok", s["id"], act, rep.get("arch"), flush=True)


def main():
    health = json.load(urllib.request.urlopen(BASE + "/api/health", timeout=60))
    print("endpoint", BASE, health["arch"], health["opencv"], flush=True)
    check_expectations()
    clips, subs, notes, t = [], [], [], 0.0
    for i, s in enumerate(SCENES):
        mp3 = os.path.join(OUT, f"{i:02d}-{s['id']}.mp3")
        if not os.path.exists(mp3):
            run([TTS, "--voice", VOICE, "--rate", "+4%", "--text", s["say"], "--write-media", mp3])
        d = dur(mp3)
        total = LEAD + d + TAIL
        clip = os.path.join(OUT, f"{i:02d}-{s['id']}.mp4")
        if os.path.exists(clip) and os.environ.get("REUSE", "1") == "1":
            pass
        elif s["kind"] == "slide":
            png = os.path.join(OUT, f"slide-{s['slide']}.png")
            run(["node", "-e", f"""
const {{ chromium }} = require('playwright');
(async()=>{{const b=await chromium.launch();const p=await b.newPage({{viewport:{{width:1920,height:1080}}}});
await p.goto('file://{HERE}/slides.html#{s["slide"]}');await p.waitForTimeout(600);await p.screenshot({{path:'{png}'}});await b.close();}})()"""],
                env={**os.environ, "NODE_PATH": NODE_PATH})
            frames = int(total * 30)
            vf = (f"scale=2400:-1,zoompan=z='min(1+0.00025*on,1.04)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                  f":d={frames}:s=1920x1080:fps=30,format=yuv420p")
            run(["ffmpeg", "-y", "-loop", "1", "-i", png, "-i", mp3, "-filter_complex",
                 f"[0:v]{vf}[v];[1:a]adelay={int(LEAD*1000)}|{int(LEAD*1000)},apad[a]", "-map", "[v]", "-map", "[a]",
                 "-t", f"{total:.3f}", "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-c:a", "aac", "-b:a", "160k", "-ar", "48000", clip])
        else:
            spec = {"base": BASE, "action": s["action"], "sample": s.get("sample"), "dur": total,
                    "out_dir": os.path.join(OUT, "raw")}
            res = json.loads(run(["node", os.path.join(HERE, "record.cjs"), json.dumps(spec)],
                                 env={**os.environ, "NODE_PATH": NODE_PATH}).strip().splitlines()[-1])
            if res.get("cut"):
                a, b = res["cut"]
                wait = b - a + 1.7
                vsrc = (f"[0:v]split[s1][s2];[s1]trim=start={res['mark']:.3f}:end={a:.3f},setpts=PTS-STARTPTS[va];"
                        f"[s2]trim=start={b:.3f},setpts=PTS-STARTPTS[vb];[va][vb]concat=n=2:v=1:a=0,")
                notes.append((t + a - res["mark"], t + a - res["mark"] + 2.5,
                              f"{wait:.0f} s on AWS Lambda, wait shortened in this video"))
            else:
                vsrc = f"[0:v]trim=start={res['mark']:.3f},setpts=PTS-STARTPTS,"
            run(["ffmpeg", "-y", "-i", res["video"], "-i", mp3, "-filter_complex",
                 f"{vsrc}fps=30,scale=1920:1080,tpad=stop_mode=clone:stop_duration=5,format=yuv420p[v];"
                 f"[1:a]adelay={int(LEAD*1000)}|{int(LEAD*1000)},apad[a]", "-map", "[v]", "-map", "[a]",
                 "-t", f"{total:.3f}", "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-c:a", "aac", "-b:a", "160k", "-ar", "48000", clip])
            os.remove(res["video"])
        # subtitles: split narration into sentence chunks spread by character count
        parts = [x.strip() for x in re.split(r"(?<=[.:;])\s+", s["say"]) if x.strip()]
        n = sum(len(x) for x in parts)
        cur = t + LEAD
        for x in parts:
            span = d * len(x) / n
            subs.append((cur, cur + span, x))
            cur += span
        clips.append(clip)
        t += total
        print(f"scene {s['id']}: {total:.1f}s (total {t:.1f}s)", flush=True)
    lst = os.path.join(OUT, "list.txt")
    open(lst, "w").write("".join(f"file '{c}'\n" for c in clips))
    joined = os.path.join(OUT, "joined.mp4")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])

    def ts(x, sep):
        ms = int(round(x * 1000))
        return f"{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d}{sep}{ms%1000:03d}"
    srt = os.path.join(OUT, "crackruler_demo.srt")
    with open(srt, "w") as f:
        for k, (a, b, x) in enumerate(subs, 1):
            f.write(f"{k}\n{ts(a, ',')} --> {ts(b, ',')}\n{x}\n\n")
    ass = os.path.join(OUT, "subs.ass")
    with open(ass, "w") as f:
        f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
                "Style: Default,Noto Sans,38,&H00E0E5E4,&H00202120,&HB0202120,0,3,10,0,2,160,160,48\n"
                "Style: Note,Noto Sans,32,&H00202120,&H0000C2F2,&H0000C2F2,0,3,10,0,9,60,60,40\n\n"
                "[Events]\nFormat: Layer, Start, End, Style, Text\n")
        for a, b, x in subs:
            cs = lambda v: f"{int(v//3600)}:{int(v//60%60):02d}:{v%60:05.2f}"
            f.write(f"Dialogue: 0,{cs(a)},{cs(b)},Default,{x}\n")
        for a, b, x in notes:
            f.write(f"Dialogue: 1,{cs(a)},{cs(b)},Note,{x}\n")
    final = os.path.join(OUT, "crackruler_demo.mp4")
    fontsdir = os.path.join(os.path.dirname(HERE), "app", "static")
    run(["ffmpeg", "-y", "-i", joined, "-vf", f"subtitles={ass}", "-c:v", "libx264", "-preset", "medium",
         "-crf", "21", "-c:a", "copy", "-movflags", "+faststart", final])
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
    total = dur(final)
    for k in range(12):
        tt = total * (k + 0.5) / 12
        run(["ffmpeg", "-y", "-ss", f"{tt:.2f}", "-i", final, "-frames:v", "1", "-vf", "scale=960:-1",
             os.path.join(OUT, "frames", f"f{k:02d}.jpg")])
    print("final", final, f"{total:.1f}s")


if __name__ == "__main__":
    main()
