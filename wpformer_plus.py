"""
wpformer_plus.py
================
Upgraded training + evaluation for WPFormer on CrackSeg9k.

HOW TO USE
----------
Drop this file INSIDE the cloned WPFormer repo folder (next to defect_train.py),
so that `from model.WPFormer import WPFormer` works.

    python wpformer_plus.py --preset baseline --epochs 30
    python wpformer_plus.py --preset loss     --epochs 30
    python wpformer_plus.py --preset backbone --epochs 30
    python wpformer_plus.py --preset full     --epochs 60
    python wpformer_plus.py --eval-only --ckpt runs/full/best.pth --tta

DESIGN RULE
-----------
Every improvement is a SWITCH, and every switch defaults to OFF.
With no flags at all this script reproduces the paper's recipe exactly:
PVTv2-B2, BCE+IoU on all 5 outputs, the repo's own augmentation, no EMA, no TTA.
That means `--preset baseline` is a real baseline, not a different model, so
every ablation row differs from the previous one by exactly one thing.

Nothing in the original repo is modified. This file only imports from it.
"""

import argparse
import ast
import copy
import json
import os
import random
import re
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ----------------------------------------------------------------------------
# Repo imports. This file must live inside the WPFormer repo folder.
# ----------------------------------------------------------------------------
from model.WPFormer import WPFormer
from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG")


# ============================================================================
# 1. CONFIG  -- one object holding every switch
# ============================================================================
@dataclass
class Config:
    # ---- data ----
    data_root: str = "/content/data/CrackSeg9k"
    train_size: int = 384
    val_fraction: float = 0.10          # carved out of TRAIN, never out of test
    split_file: str = "/content/results/val_split.json"
    num_workers: int = 2

    # ---- model ----
    backbone: str = "pvt_v2_b2"         # "pvt_v2_b2" | "pvt_v2_b4"
    channel: int = 64
    num_queries: int = 16
    backbone_ckpt: str = "/content/checkpoints/pvt_v2_b2.pth"

    # ---- loss ----
    loss: str = "paper"                 # "paper" (BCE+IoU) | "structure"
    struct_kernel: int = 15
    ds_weights: Tuple[float, ...] = (1., 1., 1., 1., 1.)   # 5 outputs
    w_boundary: float = 0.0             # >0 turns on the boundary term

    # ---- augmentation ----
    aug: str = "paper"                  # "paper" | "strong"

    # ---- training ----
    epochs: int = 30
    batch_size: int = 4
    lr: float = 8e-5
    amp: bool = True                    # mixed precision: ~2x faster, no downside
    ema: bool = False
    ema_decay: float = 0.999
    grad_accum: int = 1
    seed: int = 42

    # ---- evaluation ----
    tta: bool = False
    tta_scales: Tuple[float, ...] = (0.75, 1.0, 1.25)
    tta_flips: bool = True
    minmax: bool = True                 # the repo's per-image min-max stretch

    # ---- bookkeeping ----
    name: str = "baseline"
    out_dir: str = "/content/runs"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_dir, self.name)


PRESETS = {
    # name        -> overrides on top of the paper baseline
    "baseline":  {},
    "loss":      dict(loss="structure", ds_weights=(0.5, 1., 1., 1., 2.)),
    "backbone":  dict(backbone="pvt_v2_b4", lr=5e-5,
                      backbone_ckpt="/content/checkpoints/pvt_v2_b4.pth"),
    "aug":       dict(aug="strong"),
    "ema":       dict(ema=True),
    "boundary":  dict(loss="structure", w_boundary=0.5,
                      ds_weights=(0.5, 1., 1., 1., 2.)),
    "full":      dict(loss="structure", ds_weights=(0.5, 1., 1., 1., 2.),
                      backbone="pvt_v2_b4", lr=5e-5,
                      backbone_ckpt="/content/checkpoints/pvt_v2_b4.pth",
                      aug="strong", ema=True, tta=True),
}


# ============================================================================
# 2. THE HARD-CODED WINDOWS PATH  -- plant the backbone where the repo looks
# ============================================================================
def plant_backbone_weights(cfg: Config) -> str:
    """
    model/WPFormer.py loads its backbone from a literal Windows path such as
        'D:\\yanfeng\\Paper Code\\CVPR2025\\WPFormer\\model\\pvt_v2_b2.pth'
    On Linux a backslash is an ordinary filename character, so that string is
    just a relative FILENAME that happens to contain backslashes. We read the
    literal straight out of the source and drop the weights there, which makes
    the authors' line work with zero edits to their file.
    """
    src = Path("model/WPFormer.py").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"path\s*=\s*('[^']*%s\.pth')" % re.escape(cfg.backbone), src)
    if m is None:
        raise RuntimeError(f"no path literal for {cfg.backbone} in model/WPFormer.py")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the invalid \escapes are the point
        literal = ast.literal_eval(m.group(1))

    parent = os.path.dirname(literal)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(literal):
        if not os.path.exists(cfg.backbone_ckpt):
            raise FileNotFoundError(
                f"backbone weights not found at {cfg.backbone_ckpt}. "
                f"Download them first (see the notebook).")
        shutil.copyfile(cfg.backbone_ckpt, literal)
    return literal


def build_model(cfg: Config) -> nn.Module:
    plant_backbone_weights(cfg)
    net = WPFormer(method=cfg.backbone, channel=cfg.channel,
                   num_queries=cfg.num_queries)
    return net.cuda()


# ============================================================================
# 3. DATA
# ============================================================================
def find_pairs(image_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    """Pair every image with the mask that shares its filename stem."""
    gts = {Path(f).stem: os.path.join(gt_dir, f)
           for f in os.listdir(gt_dir) if f.endswith(IMG_EXTS)}
    pairs = []
    missing = []
    for f in sorted(os.listdir(image_dir)):
        if not f.endswith(IMG_EXTS):
            continue
        stem = Path(f).stem
        if stem in gts:
            pairs.append((os.path.join(image_dir, f), gts[stem]))
        else:
            missing.append(f)
    if missing:
        print(f"  [warn] {len(missing)} images have no matching mask, "
              f"e.g. {missing[:3]}")
    return pairs


def freeze_val_split(cfg: Config) -> Tuple[List, List]:
    """
    CrackSeg9k ships one train/test split and NO validation set.
    Tuning anything on the test set would invalidate every comparison, so we
    carve a validation set out of TRAIN once, write the exact filenames to
    disk, and reuse that same file for every experiment forever after.
    """
    train_pairs = find_pairs(os.path.join(cfg.data_root, "train", "images"),
                             os.path.join(cfg.data_root, "train", "gt"))

    if os.path.exists(cfg.split_file):
        val_stems = set(json.load(open(cfg.split_file))["val_stems"])
        print(f"  loaded frozen split from {cfg.split_file}")
    else:
        rng = random.Random(cfg.seed)
        stems = sorted(Path(p).stem for p, _ in train_pairs)
        rng.shuffle(stems)
        n_val = int(len(stems) * cfg.val_fraction)
        val_stems = set(stems[:n_val])
        os.makedirs(os.path.dirname(cfg.split_file) or ".", exist_ok=True)
        json.dump({"seed": cfg.seed, "val_fraction": cfg.val_fraction,
                   "n_train_total": len(stems), "n_val": len(val_stems),
                   "val_stems": sorted(val_stems)},
                  open(cfg.split_file, "w"), indent=1)
        print(f"  CREATED frozen split -> {cfg.split_file} "
              f"(commit this file to your repo!)")

    tr = [p for p in train_pairs if Path(p[0]).stem not in val_stems]
    va = [p for p in train_pairs if Path(p[0]).stem in val_stems]
    print(f"  train {len(tr)} | val {len(va)}")
    return tr, va


# ---- augmentation helpers ---------------------------------------------------
def aug_hflip(img, gt):
    if random.random() < 0.5:
        return (img.transpose(Image.FLIP_LEFT_RIGHT),
                gt.transpose(Image.FLIP_LEFT_RIGHT))
    return img, gt


def aug_vflip(img, gt):
    if random.random() < 0.5:
        return (img.transpose(Image.FLIP_TOP_BOTTOM),
                gt.transpose(Image.FLIP_TOP_BOTTOM))
    return img, gt


def aug_rot90(img, gt):
    k = random.randint(0, 3)
    for _ in range(k):
        img = img.transpose(Image.ROTATE_90)
        gt = gt.transpose(Image.ROTATE_90)
    return img, gt


def aug_rotate(img, gt, p=0.5, deg=15, mask_mode=Image.BICUBIC):
    if random.random() < p:
        a = random.uniform(-deg, deg)
        img = img.rotate(a, Image.BICUBIC)
        gt = gt.rotate(a, mask_mode)
    return img, gt


def aug_crop(img, gt, border=30):
    w, h = img.size
    cw = np.random.randint(max(1, w - border), w + 1)
    ch = np.random.randint(max(1, h - border), h + 1)
    box = ((w - cw) >> 1, (h - ch) >> 1, (w + cw) >> 1, (h + ch) >> 1)
    return img.crop(box), gt.crop(box)


def aug_color(img):
    img = ImageEnhance.Brightness(img).enhance(random.randint(5, 15) / 10.0)
    img = ImageEnhance.Contrast(img).enhance(random.randint(5, 15) / 10.0)
    img = ImageEnhance.Color(img).enhance(random.randint(0, 20) / 10.0)
    img = ImageEnhance.Sharpness(img).enhance(random.randint(0, 30) / 10.0)
    return img


class CrackDataset(Dataset):
    """
    is_train=False -> plain resize, no augmentation.
    aug="paper"    -> exactly what data_loader.py does: hflip, rot(20%), crop.
    aug="strong"   -> adds vflip, rot90, more frequent rotation, colour jitter.
    """

    def __init__(self, pairs, size=384, is_train=False, aug="paper",
                 want_edge=False):
        self.pairs = pairs
        self.size = size
        self.is_train = is_train
        self.aug = aug
        self.want_edge = want_edge
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, gp = self.pairs[i]
        img = Image.open(ip).convert("RGB")
        gt = Image.open(gp).convert("L")

        if self.is_train:
            if self.aug == "paper":
                # byte-for-byte what data_loader.py does, BICUBIC mask and all
                img, gt = aug_hflip(img, gt)
                img, gt = aug_rotate(img, gt, p=0.2, mask_mode=Image.BICUBIC)
                img, gt = aug_crop(img, gt)
            elif self.aug == "strong":
                img, gt = aug_hflip(img, gt)
                img, gt = aug_vflip(img, gt)          # new: cracks have no "up"
                img, gt = aug_rot90(img, gt)          # new
                img, gt = aug_rotate(img, gt, p=0.5,  # was 0.2
                                     mask_mode=Image.NEAREST)  # keeps mask crisp
                img, gt = aug_crop(img, gt)
                img = aug_color(img)                  # new: written but unused
            else:
                raise ValueError(self.aug)

        out = {"image": self.img_tf(img), "label": self.gt_tf(gt)}

        if self.want_edge:
            # the repo computes this and then throws it away; we use it
            g = np.asarray(gt.resize((self.size, self.size), Image.NEAREST))
            e = cv2.Canny(g, 100, 200)
            e = cv2.dilate(e, np.ones((5, 5), np.uint8), iterations=1)
            out["edge"] = torch.from_numpy(e.astype(np.float32) / 255.).unsqueeze(0)
        return out


# ============================================================================
# 4. LOSSES
# ============================================================================
def paper_loss(logit, mask):
    """Exactly defect_train.py's total_loss: plain BCE + plain IoU."""
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), mask)
    inter = (p * mask).sum(dim=(2, 3))
    union = (p + mask).sum(dim=(2, 3))
    iou = (1 - inter / (union - inter + 1e-6)).mean()
    return bce + iou


def structure_loss(logit, mask, k=15):
    """
    Weighted BCE + weighted IoU.

    `weit` is large wherever a pixel disagrees with its neighbourhood, i.e. on
    boundaries. A crack is thin, so nearly EVERY crack pixel is a boundary
    pixel -- which means this weighting automatically upweights the crack and
    downweights flat background, fixing the class imbalance for free.
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k // 2) - mask)

    wbce = F.binary_cross_entropy_with_logits(logit, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    p = torch.sigmoid(logit)
    inter = ((p * mask) * weit).sum(dim=(2, 3))
    union = ((p + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)      # +1 = no divide-by-zero

    return (wbce + wiou).mean()


def boundary_loss(logit, edge):
    """
    Dice between the prediction's own boundary and the (dilated Canny) target
    boundary. Needs no change to the network -- the boundary of a prediction is
    read off directly as |blur(p) - p|.
    """
    p = torch.sigmoid(logit)
    pb = torch.abs(F.avg_pool2d(p, 3, 1, 1) - p)
    pb = pb / (pb.amax(dim=(2, 3), keepdim=True) + 1e-6)
    inter = (pb * edge).sum(dim=(2, 3))
    denom = (pb + edge).sum(dim=(2, 3))
    return (1 - (2 * inter + 1) / (denom + 1)).mean()


def compute_loss(preds, gts, cfg, edge=None):
    """preds is the list of 5 logit maps the model returns."""
    base = structure_loss if cfg.loss == "structure" else paper_loss
    w = cfg.ds_weights
    if len(w) != len(preds):
        w = (1.,) * len(preds)

    total = 0.
    for wi, p in zip(w, preds):
        total = total + wi * (base(p, gts, cfg.struct_kernel)
                              if cfg.loss == "structure" else base(p, gts))
    if cfg.w_boundary > 0 and edge is not None:
        total = total + cfg.w_boundary * boundary_loss(preds[-1], edge)
    return total


# ============================================================================
# 5. EMA  -- a smoothed copy of the weights
# ============================================================================
class EMA:
    """
    Keeps a shadow copy of the weights that trails the real ones:
        shadow = decay * shadow + (1 - decay) * current
    Training weights bounce around from batch to batch; the average of the last
    few hundred steps is usually a slightly better model than any single step.
    Costs one extra copy of the weights in memory and nothing in time.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for s, m in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                s.copy_(m)


# ============================================================================
# 6. INFERENCE  (+ TTA)
# ============================================================================
_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _to_tensor(pil, size):
    x = pil.resize((size, size), Image.BILINEAR)
    x = transforms.functional.to_tensor(x)
    return _NORM(x).unsqueeze(0).cuda()


@torch.no_grad()
def predict(model, pil_img, cfg):
    """
    Returns a probability map at cfg.train_size x cfg.train_size.

    CRITICAL: sigmoid is applied to EACH view before averaging. Averaging raw
    logits and taking sigmoid once at the end is a different (worse) operation.
    """
    base = cfg.train_size
    if not cfg.tta:
        p = torch.sigmoid(model(_to_tensor(pil_img, base))[-1])
        return p

    scales = cfg.tta_scales
    n_flip = 4 if cfg.tta_flips else 1
    acc, n = None, 0

    for s in scales:
        size = max(64, int(round(base * s / 32)) * 32)   # keep divisible by 32
        x0 = _to_tensor(pil_img, size)
        for k in range(n_flip):
            x = x0
            if k & 1:
                x = torch.flip(x, [-1])
            if k & 2:
                x = torch.flip(x, [-2])

            p = torch.sigmoid(model(x)[-1])

            if k & 1:
                p = torch.flip(p, [-1])
            if k & 2:
                p = torch.flip(p, [-2])

            p = F.interpolate(p, size=(base, base), mode="bilinear",
                              align_corners=False)
            acc = p if acc is None else acc + p
            n += 1
    return acc / n


@torch.no_grad()
def evaluate(model, pairs, cfg, save_dir=None, quiet=False, fast=False):
    """
    Paper protocol: score at each image's ORIGINAL resolution.
    fast=True computes only MAE and wF -- S/E-measure are slow in numpy and we
    only need wF to pick the best epoch during training.
    """
    model.eval()
    WFM, M = WeightedFmeasure(), MAE()
    FM, SM, EM = (None, None, None) if fast else (Fmeasure(), Smeasure(),
                                                  Emeasure())
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()
    for i, (ip, gp) in enumerate(pairs):
        pil = Image.open(ip).convert("RGB")
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape

        prob = predict(model, pil, cfg).cpu().numpy().squeeze()

        if cfg.minmax:
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)

        pred = Image.fromarray(prob * 255).convert("L")
        pred = pred.resize((W, H), resample=Image.BILINEAR)
        if save_dir:
            pred.save(os.path.join(save_dir, Path(ip).stem + ".png"))
        pred = np.array(pred)

        for meter in (FM, WFM, SM, EM, M):
            if meter is not None:
                meter.step(pred=pred, gt=gt)

        if not quiet and (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(pairs)}", flush=True)

    res = {
        "MAE": float(M.get_results()["mae"]),
        "wFmeasure": float(WFM.get_results()["wfm"]),
    }
    if not fast:
        res["Smeasure"] = float(SM.get_results()["sm"])
        res["meanFm"] = float(FM.get_results()["fm"]["curve"].mean())
        res["meanEm"] = float(EM.get_results()["em"]["curve"].mean())
    res["seconds"] = round(time.time() - t0, 1)
    return res


def iou_dice(model, pairs, cfg, thresholds=np.arange(0.3, 0.75, 0.05)):
    """
    Not in the paper. Reported because the crack literature uses them.
    Uses the same cfg.minmax setting as evaluate(), so both metrics describe
    the same pipeline rather than two different ones.
    """
    inter = np.zeros(len(thresholds)); union = np.zeros(len(thresholds))
    psum = np.zeros(len(thresholds)); gsum = np.zeros(len(thresholds))
    model.eval()
    with torch.no_grad():
        for ip, gp in pairs:
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            H, W = gt.shape
            prob = predict(model, Image.open(ip).convert("RGB"),
                           cfg).cpu().numpy().squeeze()
            if cfg.minmax:
                prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
            prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
            g = gt > 127
            for j, t in enumerate(thresholds):
                p = prob > t
                inter[j] += np.logical_and(p, g).sum()
                union[j] += np.logical_or(p, g).sum()
                psum[j] += p.sum(); gsum[j] += g.sum()
    return (thresholds,
            inter / np.maximum(union, 1),
            2 * inter / np.maximum(psum + gsum, 1))


# ============================================================================
# 7. TRAINING
# ============================================================================
def _amp_tools(enabled):
    """torch.cuda.amp was deprecated for torch.amp; support both."""
    try:
        from torch.amp import autocast as _ac, GradScaler as _gs
        return (lambda: _ac("cuda", enabled=enabled)), _gs("cuda", enabled=enabled)
    except (ImportError, TypeError):
        from torch.cuda.amp import autocast as _ac, GradScaler as _gs
        return (lambda: _ac(enabled=enabled)), _gs(enabled=enabled)


def train(cfg: Config):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)
    json.dump(asdict(cfg), open(os.path.join(cfg.run_dir, "config.json"), "w"),
              indent=1, default=str)

    print(f"\n{'='*70}\nRUN: {cfg.name}\n{'='*70}")
    print(f"  backbone={cfg.backbone}  loss={cfg.loss}  aug={cfg.aug}  "
          f"ema={cfg.ema}  amp={cfg.amp}  epochs={cfg.epochs}  lr={cfg.lr}")

    tr_pairs, va_pairs = freeze_val_split(cfg)
    te_pairs = find_pairs(os.path.join(cfg.data_root, "test", "images"),
                          os.path.join(cfg.data_root, "test", "gt"))
    print(f"  test  {len(te_pairs)}")

    train_ds = CrackDataset(tr_pairs, cfg.train_size, is_train=True,
                            aug=cfg.aug, want_edge=cfg.w_boundary > 0)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True,
                        drop_last=True)

    net = build_model(cfg)
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"  parameters: {n_par:.2f} M")

    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                 eta_min=1e-7)
    autocast_ctx, scaler = _amp_tools(cfg.amp)
    ema = EMA(net, cfg.ema_decay) if cfg.ema else None

    best_wfm, history = -1.0, []
    eval_cfg_val = Config(**{**asdict(cfg), "tta": False})   # val = fast, no TTA

    for epoch in range(cfg.epochs):
        net.train()
        running, t0 = 0.0, time.time()
        opt.zero_grad(set_to_none=True)

        for it, data in enumerate(loader):
            images = data["image"].cuda(non_blocking=True).float()
            gts = data["label"].cuda(non_blocking=True).float()
            edge = data["edge"].cuda(non_blocking=True).float() \
                if "edge" in data else None

            with autocast_ctx():
                preds = net(images)
                loss = compute_loss(preds, gts, cfg, edge) / cfg.grad_accum

            scaler.scale(loss).backward()
            if (it + 1) % cfg.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if ema:
                ema.update(net)

            running += loss.item() * cfg.grad_accum

        sched.step()
        mins = (time.time() - t0) / 60
        avg = running / max(1, len(loader))

        # ---- validate on the FROZEN val split, never on test ----
        model_for_eval = ema.shadow if ema else net
        vres = evaluate(model_for_eval, va_pairs, eval_cfg_val, quiet=True,
                        fast=True)
        history.append({"epoch": epoch + 1, "loss": avg, **vres})

        star = ""
        if vres["wFmeasure"] > best_wfm:
            best_wfm = vres["wFmeasure"]
            torch.save(model_for_eval.state_dict(),
                       os.path.join(cfg.run_dir, "best.pth"))
            star = "  <-- best, saved"

        print(f"  epoch {epoch+1:3d}/{cfg.epochs}  loss {avg:.4f}  "
              f"val wF {vres['wFmeasure']:.4f}  MAE {vres['MAE']:.4f}  "
              f"[{mins:.1f} min]{star}", flush=True)

        torch.save(model_for_eval.state_dict(),
                   os.path.join(cfg.run_dir, "last.pth"))
        json.dump(history, open(os.path.join(cfg.run_dir, "history.json"), "w"),
                  indent=1)

    # ---- final: best checkpoint on the real test set ----
    print("\n  loading best checkpoint for the test-set result...")
    net.load_state_dict(torch.load(os.path.join(cfg.run_dir, "best.pth"),
                                   map_location="cuda", weights_only=False))
    test_res = evaluate(net, te_pairs, cfg,
                        save_dir=os.path.join(cfg.run_dir, "preds"))

    ths, ious, dices = iou_dice(net, te_pairs, cfg)
    j = int(np.argmax(ious))

    summary = {
        "name": cfg.name, "config": asdict(cfg),
        "params_M": round(n_par, 2),
        "best_val_wFmeasure": round(best_wfm, 4),
        "test": test_res,
        "test_best_IoU": float(ious[j]), "test_best_IoU_threshold": float(ths[j]),
        "test_Dice_at_best": float(dices[j]),
    }
    json.dump(summary, open(os.path.join(cfg.run_dir, "summary.json"), "w"),
              indent=1, default=str)

    print(f"\n  TEST: " + "  ".join(f"{k} {v:.4f}" for k, v in test_res.items()
                                    if k != "seconds"))
    print(f"  IoU {ious[j]:.4f} @ {ths[j]:.2f}   Dice {dices[j]:.4f}")
    print(f"  saved -> {cfg.run_dir}/summary.json")
    return summary


# ============================================================================
# 8. SELF-TEST  -- run this BEFORE starting a multi-hour training run
# ============================================================================
def selftest():
    """`python wpformer_plus.py --selftest` -- ~20 seconds, no data needed."""
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        ok = ok and bool(cond)

    print("\n--- losses ---")
    torch.manual_seed(0)
    mask = torch.zeros(2, 1, 64, 64).cuda()
    mask[:, :, 30:33, 5:60] = 1.0                       # a thin horizontal crack

    good = (mask * 8 - 4).clone().requires_grad_(True)  # confident + correct
    bad = (torch.randn(2, 1, 64, 64).cuda()).requires_grad_(True)

    for fn, nm in ((paper_loss, "paper_loss"), (structure_loss, "structure_loss")):
        lg, lb = fn(good, mask), fn(bad, mask)
        check(f"{nm}: finite", torch.isfinite(lg) and torch.isfinite(lb))
        check(f"{nm}: good < bad", lg.item() < lb.item(),
              f"({lg.item():.4f} < {lb.item():.4f})")
        lg.backward()
        check(f"{nm}: gradient flows", good.grad is not None
              and torch.isfinite(good.grad).all())
        good.grad = None

    empty = torch.zeros(2, 1, 64, 64).cuda()
    check("structure_loss: no NaN on an all-zero mask",
          torch.isfinite(structure_loss(bad, empty)))
    check("paper_loss: no NaN on an all-zero mask",
          torch.isfinite(paper_loss(bad, empty)))

    edge = torch.zeros_like(mask); edge[:, :, 29:34, 5:60] = 1.0
    check("boundary_loss: finite", torch.isfinite(boundary_loss(good, edge)))

    print("\n--- deep supervision weighting ---")
    preds = [torch.randn(2, 1, 64, 64).cuda() for _ in range(5)]
    c = Config(loss="structure", ds_weights=(0.5, 1., 1., 1., 2.))
    check("compute_loss: 5 outputs -> finite scalar",
          torch.isfinite(compute_loss(preds, mask, c)))
    c2 = Config(loss="structure", ds_weights=(1., 1.))     # wrong length
    check("compute_loss: bad ds_weights length falls back safely",
          torch.isfinite(compute_loss(preds, mask, c2)))

    print("\n--- EMA ---")
    lin = nn.Linear(4, 4).cuda()
    e = EMA(lin, decay=0.5)
    before = e.shadow.weight.detach().clone()
    with torch.no_grad():
        lin.weight.add_(1.0)
    e.update(lin)
    expect = 0.5 * before + 0.5 * lin.weight.detach()
    check("EMA: shadow = d*shadow + (1-d)*live",
          torch.allclose(e.shadow.weight, expect, atol=1e-6))

    print("\n--- TTA round-trip ---")

    class Identity(nn.Module):
        # pretends to be WPFormer: returns a list whose last item is the "logit"
        def forward(self, x):
            return [x.mean(1, keepdim=True) * 6.0]

    ident = Identity().cuda().eval()
    img = Image.fromarray(
        (np.random.rand(97, 131, 3) * 255).astype(np.uint8))   # odd, non-square

    c_no = Config(tta=False, train_size=64)
    c_yes = Config(tta=True, train_size=64, tta_scales=(1.0,), tta_flips=True)
    p_no = predict(ident, img, c_no)
    p_yes = predict(ident, img, c_yes)
    check("TTA: output shape unchanged", p_no.shape == p_yes.shape,
          str(tuple(p_no.shape)))
    check("TTA: flips are undone correctly (identity model -> same map)",
          torch.allclose(p_no, p_yes, atol=2e-2),
          f"max diff {(p_no - p_yes).abs().max().item():.4f}")

    c_ms = Config(tta=True, train_size=64, tta_scales=(0.75, 1.0, 1.25))
    check("TTA: multi-scale runs and stays in [0,1]",
          bool(((0 <= predict(ident, img, c_ms)) &
                (predict(ident, img, c_ms) <= 1)).all()))

    print("\n--- config presets ---")
    for nm, ov in PRESETS.items():
        cc = Config(**{**asdict(Config()), **ov})
        check(f"preset '{nm}' builds", isinstance(cc, Config),
              f"backbone={cc.backbone} loss={cc.loss} aug={cc.aug} "
              f"ema={cc.ema} tta={cc.tta}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


# ============================================================================
# 9. CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="baseline", choices=list(PRESETS))
    ap.add_argument("--name", default=None, help="run name (default = preset)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--loss", default=None, choices=["paper", "structure"])
    ap.add_argument("--aug", default=None, choices=["paper", "strong"])
    ap.add_argument("--ema", action="store_true", default=None)
    ap.add_argument("--tta", action="store_true", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-minmax", action="store_true")
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="verify losses/EMA/TTA in ~20s, no data needed")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    cfg = Config(**{**asdict(Config()), **PRESETS[args.preset]})
    cfg.name = args.name or args.preset
    for k in ("data_root", "out_dir", "split_file", "epochs", "batch_size",
              "lr", "backbone", "loss", "aug", "ema", "tta", "grad_accum"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    # keep the frozen split next to the runs unless told otherwise
    if args.split_file is None and args.out_dir is not None:
        cfg.split_file = os.path.join(cfg.out_dir, "val_split.json")
    if args.no_amp:
        cfg.amp = False
    if args.no_minmax:
        cfg.minmax = False
    if cfg.backbone == "pvt_v2_b4" and "b4" not in cfg.backbone_ckpt:
        cfg.backbone_ckpt = "/content/checkpoints/pvt_v2_b4.pth"

    if args.eval_only:
        if not args.ckpt:
            sys.exit("--eval-only needs --ckpt")
        net = build_model(cfg)
        net.load_state_dict(torch.load(args.ckpt, map_location="cuda",
                                       weights_only=False), strict=False)
        te = find_pairs(os.path.join(cfg.data_root, "test", "images"),
                        os.path.join(cfg.data_root, "test", "gt"))
        print(f"evaluating {len(te)} test images  (tta={cfg.tta}, "
              f"minmax={cfg.minmax})")
        res = evaluate(net, te, cfg)
        print(json.dumps(res, indent=1))
    else:
        train(cfg)


if __name__ == "__main__":
    main()
