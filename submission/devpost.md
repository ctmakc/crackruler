# Devpost submission text (CrackRuler)

**Project name:** CrackRuler

**Tagline (elevator pitch):** Concrete crack widths in millimetres from one phone photo and a printed marker. An OpenCV 5 inspection agent on AWS Lambda (Graviton) that asks a person when the evidence is weak.

**Built with:** opencv, python, onnx, pytorch, numpy, aws-lambda, amazon-s3, amazon-cloudwatch, aws-cloudformation, aws-x-ray, aws-graviton, javascript, html5, css3

**Links:** Try it out: https://hwytllvqjarqppbcobgqs25lia0hywlx.lambda-url.us-east-1.on.aws/ · Code: https://github.com/ctmakc/crackruler · Technical report: https://github.com/ctmakc/crackruler/blob/main/docs/TECHNICAL_REPORT.md

**Video:** https://www.youtube.com/watch?v=8CUrgB_a84I

**Gallery images:** docs/architecture.png, docs/agent-loop.png, docs/example-annotated.jpg, docs/failure-cases.jpg, UI screenshot (result view)

---

## Inspiration
Crack width decides what happens to a concrete wall, a parking deck or a bridge pier. Inspectors read it with a plastic comparator card, write the number on a form and come back months later to see whether the crack has grown. A phone photo of the same crack is easy to take, but it has no scale, it is tilted, and nobody knows whether it was sharp enough to show a 0.2 mm crack.

## What it does
You print a 50 mm ArUco marker, put it flat next to the crack, take a photo and pick the exposure class. CrackRuler returns:
- every crack's length and its median, 95th-percentile and maximum width in millimetres, with an uncertainty;
- the smallest width this particular photo can resolve;
- a verdict against the ACI 224R-01 Table 4.1 guideline for that exposure: log, monitor, engineer review, retake (closer or steadier), or human review;
- an annotated evidence image, a comparator-card view of the widths, and the full agent trace.

## How we built it
- **OpenCV 5 perception tools.** `aruco.ArucoDetector` with sub-pixel corners fixes scale; a homography (`getPerspectiveTransform`, `warpPerspective`) makes the surface fronto-parallel at the marker's native pixel density. The marker's black/white edge doubles as a blur probe: the 10-90 % edge rise gives the optical sigma of this photo.
- **Two detectors.** A 1.56 M-parameter U-Net, trained in PyTorch on the CC BY 4.0 Concrete Crack Segmentation Dataset and exported to ONNX, runs inside the OpenCV 5 DNN engine (`dnn.readNetFromONNX`, PyTorch parity 9e-5). A classical multi-scale Hessian ridge filter built from `Sobel` second derivatives gives an independent second opinion.
- **Width measurement.** Centreline by `ximgproc.thinning`, grey-level profiles across the crack sampled at 0.25 px with `remap` on the full-resolution rectified image, full width at half depth, then an erf blur model inverted with the sigma measured on the marker. Widths the optics cannot resolve are reported as upper bounds.
- **The agent.** A deterministic planner reads each tool's output and picks the next tool. Blur, a missing marker, disagreement between the detectors, or too few pixels across the crack each change its next step; every uncertain case ends with a person. Each step is logged with its observation and timing.
- **AWS.** One Lambda function (Python 3.12, arm64 Graviton, 512 MB) behind a Lambda Function URL serves the UI and the API. To fit 512 MB, the U-Net runs on overlapping 512 px windows; on the 69 test photos the result is the same as a single pass. Reports and annotated images go to a private, encrypted S3 bucket that expires them after 7 days; the uploaded photo is never stored. CloudWatch gets structured logs, Embedded Metric Format metrics, a dashboard and an error alarm; X-Ray traces every call. One CloudFormation template deploys it, and the arm64 zip builds on any machine with `pip --platform` (no Docker). An AWS Budget alert guards the account.

## Evaluation (all scripts and outputs in the repo)
- 69 held-out real photos: U-Net pixel F1 0.937 (2 px tolerance), IoU 0.757; classical detector F1 0.930.
- 1000 + 1000 crack / no-crack patches: 99.7 % (U-Net) and 99.3 % (classical); the patches come from the same photo family, so this is not independent for the U-Net.
- Width in millimetres, **synthetic** scenes (real concrete backgrounds, synthetic crack and marker, 0-30° tilt, blur, JPEG; 84 scenes): mean error at most 0.045 mm once a crack is about 2.4 px across or wider; thinner cracks are overestimated, which pushes the verdict towards review; 0.1 mm cracks were not detected at 5-12 px/mm.

## Challenges we ran into
Blur makes thin cracks look wider, so a naive width is biased exactly where the decision matters. Measuring the blur on the marker itself and inverting it fixed most of the bias and also told the agent when a photo cannot resolve the guideline at all.

Our new AWS account allows 512 MB per Lambda function. The single-pass U-Net needed about 650 MB, so we split it into overlapping windows clipped at the image edge; peak memory on Lambda is now 446 MB and the test-set F1 did not change. The price is speed: an inspection takes about 23 s on arm64 and 11 s on an x86 twin we deployed for comparison. The classical OpenCV tools run at about the same speed on both; the U-Net step is 2.5 times slower with the stock aarch64 wheel. The numbers are in the report, and the x86 twin stays online for comparison.

## Accomplishments that we're proud of
The whole chain is honest about its limits: synthetic results are labelled, widths below resolution are upper bounds, unconfirmed detections are shown as such, and the system asks for a closer photo with a concrete factor computed from the measured pixel density.

## What we learned
Most of the accuracy comes from geometry and optics: scale, plane and blur. The network's job is to find where the crack runs; the measurement happens on the image itself.

## What's next
Field photos with comparator-card readings to validate metric accuracy on real cracks; repeat visits of the same crack to track growth; an MCP tool interface so other agents can call the inspection tools.

## Responsible use
ACI 224R-01 Table 4.1 is a design guideline for new construction under service loads; CrackRuler uses it to route a finding and never certifies a structure as safe. Photos are processed in memory and not stored; reports expire after 7 days. Training data is CC BY 4.0 and attributed. The code was written with the help of an AI coding assistant; no language model runs in the product.
