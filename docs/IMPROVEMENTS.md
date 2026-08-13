# Improvements

Every change below is a switch in `wpformer_plus.py`, and **every switch defaults to off**.
With no flags the script reproduces the paper's recipe exactly, so `--preset baseline` is a
genuine baseline and each ablation row differs from the one above it by exactly one thing.

Estimated gains are typical magnitudes for this kind of change on thin-structure
segmentation. **They are not measured on this setup** — treat them as priorities, not
promises.

| # | Change | Work | Retrain? | Est. ΔwFβ | Risk |
|---|---|---|---|---|---|
| 1 | TTA + multi-scale | 1 h | no | +0.5 – 1.5 | very low |
| 2 | PVTv2-B2 → B4 | 15 min | yes | +1.0 – 2.0 | low |
| 3 | Structure loss | 30 min | yes | +0.8 – 1.8 | low |
| 4 | Photometric aug + vflip/rot90 | 20 min | yes | +0.3 – 0.8 | very low |
| 5 | EMA weights | 30 min | yes | +0.2 – 0.6 | very low |
| 6 | Deep-supervision reweighting | 10 min | yes | +0.1 – 0.4 | low |
| 7 | Boundary loss on the discarded edge map | 45 min | yes | unknown | medium |
| — | Drop per-image min–max at eval | 5 min | no | unknown | test it |

Combined, expect **+2.5 to +3.5%** — they do not stack linearly.

---

## 1 · Structure loss

### What is wrong with the original

```python
def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce = nn.BCELoss()(pred, mask)      # every pixel counts the same
    ...
```

Every pixel counts the same. In a 384×384 crack photo — 147,456 pixels — perhaps 2,000 are
crack. **98.6% of the image is background.** A model that learns only "say no crack" scores
98.6% and is useless, and nothing in the loss tells it otherwise.

### The fix

```python
weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k//2) - mask)
```

Three steps: blur the true mask with a 15×15 window; ask where the blur disagrees with the
original; turn disagreement into a weight from 1 to 6.

| Location | blurred | true | difference | **weight** |
|---|---|---|---|---|
| flat background | 0.0 | 0 | 0.0 | **1.0** |
| interior of a large blob | 1.0 | 1 | 0.0 | **1.0** |
| edge of a blob | 0.5 | 1 | 0.5 | **3.5** |
| **3-pixel-wide crack** | ~0.2 | 1 | 0.8 | **5.0** |

A crack pixel is worth 5× a background pixel, automatically.

The elegant part: because a crack is thin, **almost every crack pixel is also an edge pixel**.
A formula designed to emphasise boundaries ends up emphasising the entire crack. The
class-imbalance correction falls out of the object's geometry rather than a tuned constant.

### Why this over Focal Tversky

The project plan originally called for Focal Tversky. It is a reasonable third ablation row,
but it introduces α, β and γ that must be found by trial — training runs we do not have time
for. Structure loss has one hyper-parameter (`k`) with a sensible default. Try `k = 15` and
`k = 31` as two rows; smaller suits thinner cracks.

### Two repairs in the same function

```python
wiou = 1 - (inter + 1) / (union - inter + 1)
```

The original has no `+1`. When a prediction and mask are both empty it computes `0/0` → NaN →
the run dies silently. Also, the original applies `sigmoid` then `BCELoss`; we use
`binary_cross_entropy_with_logits`, which fuses both stably and cannot take `log(0)`.

---

## 2 · PVTv2-B2 → B4

The project plan budgeted days for "debug shape mismatches, adapt the channel and feature-map
interfaces". None of that is required.

Every PVTv2 variant emits the **same** channel widths `[64, 128, 320, 512]`; only depth
changes:

| Variant | depths |
|---|---|
| B1 | `[2, 2, 2, 2]` |
| B2 (paper) | `[3, 4, 6, 3]` |
| B3 | `[3, 4, 18, 3]` |
| **B4** | `[3, 8, 27, 3]` |
| B5 | `[3, 6, 40, 3]` |

B4 is **deeper, not wider**, so `Translayer1_1` … `Translayer4_1` need no modification. And
`WPFormer.__init__` already contains the branch:

```python
if method == "pvt_v2_b2":
    self.backbone = pvt_v2_b2()
else:
    self.backbone = pvt_v2_b4()
```

Learning rate drops **8e-5 → 5e-5**: a deeper pretrained encoder is more delicate, and pushing
it hard in the first epochs damages the ImageNet features it arrived with.

**Skip Swin and DINOv2.** Days of interface work for no advantage over B4 here.

If B4 exhausts T4 memory, use `--batch-size 2 --grad-accum 2`: two images at a time, weights
updated every second batch, so the *effective* batch stays 4 with half the memory.

---

## 3 · Augmentation

### Already in the repo — do not duplicate

`data_loader.py` already applies, for CrackSeg9k: horizontal flip; rotation ±15° **but only
20% of the time**; random crop.

### Added

```python
img, gt = aug_vflip(img, gt)             # new
img, gt = aug_rot90(img, gt)             # new
img, gt = aug_rotate(img, gt, p=0.5)     # was 0.2
img = aug_color(img)                     # new -- already written, never called
```

**Why vertical flip and rot90 are valid here.** An upside-down cat is an unnatural image and
training on it can hurt. A crack has **no correct orientation** — one running top-to-bottom is
as real as one running left-to-right. That yields 8 valid variants of every photo.

**`colorEnhance()` was already implemented in the repo and simply never called** — brightness,
contrast, colour and sharpness jitter, sitting unused. Enabling it is one line, and crack
photographs vary enormously in lighting and surface type.

**Deliberately not enabled: `randomPeper()`.** Also present and unused, it adds salt-and-pepper
noise to the **ground-truth mask**. On a 3-pixel-wide crack that means randomly deleting and
inventing crack pixels — corrupting the labels.

**Interpolation detail.** Rotating a mask requires choosing how to fill in-between pixels. The
repo uses `BICUBIC`, which invents smooth grey values and leaves a soft halo around a mask that
should be binary. The strong preset uses `NEAREST`. The baseline keeps `BICUBIC` — "fixing" it
there would mean the baseline is no longer the paper.

---

## 4 · Boundary loss on a signal the repo discards

From `data_loader.py`:

```python
gt = np.asarray(gt)
edge = cv2.Canny(gt, 100, 200)
edge = cv2.dilate(edge, np.ones((5, 5), np.uint8), iterations=1)
return {'image': image, 'label': gt, "edge": edge}
```

`defect_train.py` reads only `data['image']` and `data['label']`. **The edge map is computed
every iteration and thrown away.** The boundary-aware loss the project plan asks for already
has its target in the pipeline.

Our term needs **no change to the network**:

```python
def boundary_loss(logit, edge):
    p = torch.sigmoid(logit)
    pb = torch.abs(F.avg_pool2d(p, 3, 1, 1) - p)   # the prediction's own outline
    pb = pb / (pb.amax(dim=(2, 3), keepdim=True) + 1e-6)
    ...  # dice against the target outline
```

Same "blur and subtract" trick as the structure loss — that is how you read an outline off any
map — then push the prediction's outline towards the true one.

**Off by default** (`w_boundary = 0.0`). It is the most experimental change here and is
untested; treat it as an ablation row, not a guaranteed gain.

---

## 5 · EMA

```
shadow = 0.999 * shadow + 0.001 * current
```

Training weights jiggle: every batch of 4 images tugs them in whatever direction those 4
images suggest. When training stops you are wherever the *last* batch left you, which may be
an unlucky spot.

The shadow copy trails behind, absorbing 0.1% of the live weights each step, so it sits near
the average of the last few hundred steps. A car on a bumpy road bounces; a long-exposure
photograph shows the smooth path it actually travelled. EMA is the long exposure.

One extra copy of the weights in memory, nothing in time.

---

## 6 · TTA and multi-scale

No training at all — this can be run against the authors' released checkpoint immediately.

Show the model **12 versions** of each photo (3 scales × 4 flips) and average the answers.

```python
for s in scales:                        # 0.75, 1.0, 1.25
    for k in range(4):                  # identity, hflip, vflip, both
        x = flip(input, k)
        p = torch.sigmoid(model(x)[-1])    # sigmoid HERE, per view
        p = flip(p, k)                     # flip the ANSWER back
        acc = acc + p
return acc / n
```

Two silent failure modes, both handled:

1. **Flip the answer back.** Flip the photo left-right and the predicted crack comes out on
   the wrong side; averaging without undoing the flip blurs every crack into mush. Flipping is
   its own inverse, so the same operation undoes it.
2. **Sigmoid each view *before* averaging.** Averaging raw logits and applying sigmoid once at
   the end is a different mathematical operation with worse results. No error is raised — just
   weaker numbers.

`--selftest` verifies the round-trip by running TTA against a model that returns its input and
checking the output comes back identical.

Costs 12× inference. Irrelevant for 395 images.

---

## 7 · Deep-supervision reweighting

The model produces five predictions and the original grades them equally:

```python
for i in range(len(predictions_mask)):
    mask_losses = mask_losses + total_loss(predictions_mask[i], gts)
```

But `predictions_mask[-1]` — the sum S1+S2+S3 — is **the only one used at test time**. The
others are teaching aids. So:

```python
ds_weights = (0.5, 1., 1., 1., 2.)
```

The final answer counts double; `S0`, the earliest and roughest guess, counts half.

---

## Not accuracy, but decisive

### Mixed precision

The original recipe is 7243 ÷ 4 = 1,810 steps/epoch × 60 = **~108,000 steps**, roughly
**8–10 hours per run** on a T4 — times five runs, against Colab's disconnects. This, not the
algorithms, is what threatens the schedule.

Mixed precision does most arithmetic in 16-bit: half the data movement, dedicated hardware,
**roughly 2× faster at no accuracy cost**. The `GradScaler` handles the one hazard — 16-bit
cannot represent very small values, so tiny gradients would round to zero — by scaling the
loss up before backprop and dividing back after.

Combined with 30-epoch ablations, ~9 h per experiment becomes ~2 h. A comparison is fair when
both sides share a budget; that budget need not be the paper's.

### The frozen validation split

CrackSeg9k ships train and test and **no validation set**. Choosing the best epoch by test
score is cheating, and visibly so.

10% of *train* is carved off once with a fixed seed and the exact filenames written to JSON.
Best epoch chosen on validation; IoU threshold chosen on validation; test touched exactly once.

**Commit that JSON.** It is what makes three team members' numbers comparable — independent
random splits would make the ablation table meaningless.

### The min–max question

`defect_test.py` stretches every prediction so its darkest pixel becomes 0 and its brightest 1.
On an image where the model is *correctly* unsure, that turns quiet uncertainty into a
confident false positive.

It is standard practice in the saliency literature and may well be helping. It costs five
minutes and no retraining to find out, and either answer is a legitimate ablation row:
`--no-minmax`.
