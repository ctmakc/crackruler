"""Narration and scene list for the submission video. Numbers here must match a live run:
make_video.py checks every 'expect' value against the endpoint's actual report before recording."""

SCENES = [
    {"id": "title", "kind": "slide", "slide": "title",
     "say": "CrackRuler. I'm Maksym Stepanenko from INNOVA CONSULT in Ottawa. Inspectors decide what happens "
            "to a concrete wall or a bridge deck by the width of its cracks. Today they read that width with a "
            "plastic card and write it on a form. CrackRuler gets it in millimetres from one phone photo and a printed marker."},
    {"id": "ui", "kind": "live", "action": "intro",
     "say": "This is the live app, running on AWS Lambda on Graviton. You print the fifty millimetre ArUco marker, "
            "lay it flat next to the crack, take a photo, and pick the exposure. Each exposure carries its guideline "
            "width from ACI 224R."},
    {"id": "s035", "kind": "live", "action": "run", "sample": "synthetic-0.35mm.jpg",
     "expect": {"action": "monitor", "median": 0.36},
     "say": "Here is a synthetic test scene: a real concrete background and a crack exactly 0.35 millimetres wide. "
            "The agent finds the marker, removes the camera tilt, finds the crack with a U-Net running in the OpenCV 5 "
            "DNN engine, checks it with a classical ridge detector, and measures across the crack on the full resolution "
            "image. Median width 0.36 millimetres. The 95th percentile sits just above the 0.3 millimetre guideline for "
            "humid exposure, inside the uncertainty band, so the verdict is monitor."},
    {"id": "trace", "kind": "live", "action": "trace", "sample": "synthetic-0.35mm.jpg",
     "say": "Every step is in the trace: what each OpenCV tool saw, how long it took, and why the agent chose the next one."},
    {"id": "tilt", "kind": "live", "action": "run", "sample": "synthetic-0.60mm-tilted.jpg",
     "expect": {"action": "engineer_review", "median": 0.61},
     "say": "Tilted twenty-eight degrees, true width 0.6 millimetres, measured 0.61. That is clearly over the guideline, "
            "so the case goes to an engineer, and the report flags that a person must review it."},
    {"id": "far", "kind": "live", "action": "run", "sample": "synthetic-far-away.jpg",
     "expect": {"action": "retake"},
     "say": "When a photo cannot support a measurement, the agent says so. Shot from too far away, the smallest width "
            "this photo can resolve is wider than the guideline, so it asks you to move about 1.8 times closer."},
    {"id": "blur", "kind": "live", "action": "run", "sample": "synthetic-blurred.jpg",
     "expect": {"action": "retake"},
     "say": "Out of focus: the marker's own black and white edge shows the blur, and the agent asks for a steadier shot."},
    {"id": "real", "kind": "live", "action": "run", "sample": "real-191.jpg",
     "expect": {"action": "human_review"},
     "say": "On a real photo without a marker it still finds the crack, but it will not invent millimetres. "
            "Widths stay in pixels, and a person decides."},
    {"id": "arch", "kind": "slide", "slide": "architecture",
     "say": "The whole system is one arm64 Lambda function behind a function URL, built and deployed with one "
            "CloudFormation template. Reports go to a private S3 bucket that deletes them after seven days, and the photo "
            "itself is never stored. CloudWatch gets structured logs, metrics, a dashboard and an error alarm."},
    {"id": "loop", "kind": "slide", "slide": "agent-loop",
     "say": "The agent is a small deterministic planner. Blur, a missing marker, disagreement between the two detectors, "
            "or too few pixels across the crack each change its next step, and every uncertain case ends with a person."},
    {"id": "results", "kind": "slide", "slide": "results",
     "say": "On 69 held-out real photos the U-Net reaches a pixel F1 of 0.94. On 84 synthetic scenes with known widths, "
            "once a crack is at least two and a half pixels across, the mean width error stays under five hundredths of a "
            "millimetre. Thinner cracks are overestimated, which pushes the verdict towards review, and 0.1 millimetre "
            "cracks are not detected yet."},
    {"id": "close", "kind": "slide", "slide": "close",
     "say": "Metric accuracy is shown on synthetic scenes so far; field photos with comparator card readings are the next "
            "step. The guideline routes findings and never certifies a structure. Code, data split, evaluation and the "
            "template are in the public repository."},
]
