"""Train a small U-Net for crack segmentation on the TRAIN split of the
Concrete Crack Segmentation Dataset (Özgenel 2019, CC BY 4.0) and export it to ONNX
for inference with the OpenCV 5 DNN engine. Training uses PyTorch; inference does not."""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eval.common import seg_pairs, load_split, load_pair, pixel_scores, summarise

SEED = 26
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(OUT, exist_ok=True)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, c=(16, 32, 64, 128, 192)):
        super().__init__()
        self.enc = nn.ModuleList([block(3, c[0])] + [block(c[i], c[i + 1]) for i in range(len(c) - 1)])
        self.dec = nn.ModuleList([block(c[i + 1] + c[i], c[i]) for i in reversed(range(len(c) - 1))])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for d in self.dec:
            s = skips.pop()
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = d(torch.cat([x, s], 1))
        return self.head(x)


def augment(img, gt, crop=int(os.environ.get("CROP", 352))):
    s = random.uniform(0.5, 1.3)
    img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA if s < 1 else cv.INTER_LINEAR)
    gt = cv.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)
    h, w = gt.shape
    for _ in range(10):
        y, x = random.randint(0, h - crop), random.randint(0, w - crop)
        g = gt[y:y + crop, x:x + crop]
        if g.any() or random.random() < 0.25:
            break
    im = img[y:y + crop, x:x + crop].astype(np.float32)
    k = random.randint(0, 3)
    im, g = np.rot90(im, k), np.rot90(g, k)
    if random.random() < 0.5:
        im, g = im[:, ::-1], g[:, ::-1]
    im = im * random.uniform(0.7, 1.3) + random.uniform(-30, 30)       # brightness / contrast
    if random.random() < 0.3:
        im = cv.GaussianBlur(np.ascontiguousarray(im), (0, 0), random.uniform(0.5, 1.5))
    if random.random() < 0.3:
        im = im + np.random.normal(0, random.uniform(2, 8), im.shape)
    if random.random() < 0.2:                                            # grey-ish surfaces
        gray = im.mean(2, keepdims=True); im = gray + (im - gray) * random.uniform(0, 0.6)
    im = np.clip(im, 0, 255)[:, :, ::-1] / 255.0                         # BGR -> RGB, as blobFromImage(swapRB=True)
    return np.ascontiguousarray(im.transpose(2, 0, 1)), np.ascontiguousarray(g[None].astype(np.float32))


def main(steps=6000, bs=int(os.environ.get("BS", 8))):
    split = load_split()
    pairs = {s: (ip, mp) for s, ip, mp in seg_pairs()}
    train = [load_pair(*pairs[s]) for s in split["train"]]
    val = [load_pair(*pairs[s]) for s in split["val"]]
    print("loaded", len(train), len(val), flush=True)
    net = UNet().to(DEV)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    pos_w = torch.tensor([3.0], device=DEV)
    best, log = -1, []
    t0 = time.time()
    for step in range(1, steps + 1):
        net.train()
        batch = [augment(*random.choice(train)) for _ in range(bs)]
        x = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(DEV)
        y = torch.from_numpy(np.stack([b[1] for b in batch])).float().to(DEV)
        with torch.autocast(DEV, dtype=torch.float16):
            lo = net(x)
        lo = lo.float()
        p = torch.sigmoid(lo)
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
        loss = F.binary_cross_entropy_with_logits(lo, y, pos_weight=pos_w) + dice
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 500 == 0 or step == steps:
            net.eval(); rows = []
            with torch.no_grad():
                for img, gt in val:
                    h, w = gt.shape
                    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
                    im = cv.copyMakeBorder(img, 0, ph, 0, pw, cv.BORDER_REFLECT)[:, :, ::-1] / 255.0
                    t = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))[None].float().to(DEV)
                    with torch.autocast(DEV, dtype=torch.float16):
                        lo_v = net(t)
                    pr = torch.sigmoid(lo_v.float())[0, 0, :h, :w].cpu().numpy()
                    del lo_v, t
                    rows.append(pixel_scores((pr > 0.5).astype(np.uint8), gt))
            s = summarise(rows); s.update(step=step, loss=round(loss.item(), 4), min=round((time.time() - t0) / 60, 1))
            log.append(s); print(json.dumps(s), flush=True)
            if s["f1"] > best:
                best = s["f1"]; torch.save(net.state_dict(), os.path.join(OUT, "best.pt"))
    json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"), indent=1)
    net.load_state_dict(torch.load(os.path.join(OUT, "best.pt"), weights_only=True)); net.eval().cpu()
    dummy = torch.zeros(1, 3, 256, 256)
    onnx_path = os.path.join(os.path.dirname(OUT), "..", "models", "crack_unet.onnx")
    torch.onnx.export(net, dummy, onnx_path, input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}})
    print("exported", onnx_path, "best val f1", best)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6000)
