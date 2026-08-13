# Reproduction notes

The original repository was written for **Python 3.7.13 / PyTorch 1.11.0 / CUDA 11.3.1** on the
author's Windows machine. Colab today runs Python 3.1x and PyTorch 2.x on Linux. Four things
had to be dealt with, and **none of them required editing a file in the authors' repository.**

Reference: [`fengyan-cv/WPFormer`](https://github.com/fengyan-cv/WPFormer), pinned to commit
[`83a33bb`](https://github.com/fengyan-cv/WPFormer/commit/83a33bbf5ed96dff069e9d58f5f3e0c464bae446).

---

## 1 · Absolute Windows paths

`model/WPFormer.py` line 267:

```python
path = 'D:\yanfeng\Paper Code\CVPR2025\WPFormer\model\pvt_v2_b2.pth'
save_model = torch.load(path)
```

The author's own drive letter shipped with the release.

**The trick.** On Windows `\` separates directories. On Linux it is an **ordinary character in
a filename**, as ordinary as a letter. So on Linux that string is not a path at all — it is a
single relative *filename* that happens to contain backslashes.

Which means the file does not need editing. Create a file with exactly that name and the line
works as written:

```python
m = re.search(r"path\s*=\s*('[^']*pvt_v2_b2\.pth')", source)
literal = ast.literal_eval(m.group(1))     # resolves escapes exactly as Python's parser does
shutil.copyfile(downloaded_weights, literal)
```

`ast.literal_eval` rather than retyping the string: every escape in that path (`\y`, `\P`,
`\C`, `\W`, `\m`, `\p`) is *invalid*, so Python preserves each backslash literally. Letting
Python's own machinery resolve it makes the name correct by construction.

## 2 · The string-concatenation trap

`defect_test.py` builds its file list without `os.path.join`:

```python
test_image_root = os.path.join(file_dir, dataset_name + "\\test\\images\\")
images = [test_image_root + f for f in os.listdir(test_image_root)]
```

On Linux that root resolves to two components — a directory `.\datasets\` containing a
directory `CrackSeg9k\test\images\`.

`os.listdir` on it works fine. But `root + f` appends to **the directory's own name**, yielding
`CrackSeg9k\test\images\a.jpg`, which is a *sibling* of that directory rather than a file
inside it.

So running the original script untouched requires creating both: the directory, so the listing
succeeds, and one flat symlink per image under the concatenated name, so the reads succeed.
Notebook 01 does this in its optional final section.

## 3 · Library drift

| Problem | Fix |
|---|---|
| `timm.models.layers` and `timm.models.registry` moved in modern `timm` | alias modules created at the old import paths, forwarding to the new ones |
| `defect_test.py` imports `mmcv.cnn.get_model_complexity_info` at module level and never calls it | a stub `mmcv` package providing just that function |
| PyTorch ≥ 2.6 flipped `torch.load` to `weights_only=True`; the checkpoints predate it | `torch.load` wrapped to restore the old default |

The same shims are written to a `sitecustomize.py` on `PYTHONPATH` so a subprocess
`python defect_test.py` inherits them.

> The `weights_only=False` restoration un-pickles the checkpoint, which executes code from the
> file. That is true of every research checkpoint distributed this way; these come from the
> Google Drive links in the official README.

## 4 · Data integrity checks

Two guards run before any measurement, and both matter for a *faithful* reproduction.

**Image count.** The paper states CrackSeg9k has **7243 train / 395 test**. A different count
means a different dataset version, and the numbers stop being comparable to Table 1.

**Filename pairing.** The original pairs images with masks by alphabetical order and never
verifies the pairing:

```python
images = sorted(os.listdir(image_folder))
gts    = sorted(os.listdir(gt_folder))
# assumes images[5] corresponds to gts[5]
```

If images are `img_1.jpg … img_10.jpg` while masks are `1.png … 10.png`, sorting orders them
differently and photo #5 is scored against mask #12 — for all 395 images, with **no error
raised**. You would get plausible-looking bad numbers and blame the model. So filename stems
are compared explicitly and any mismatch is reported loudly.

---

## Verified facts about the model

Established by reading the source rather than assuming:

- `forward()` sets `image_shape = x.size()[2:]`, so all outputs are at **full input
  resolution**.
- `class_embed = nn.Linear(channel, 1)`, so `semantic_inference` yields **single-channel
  logits** — sigmoid is applied outside the network, in the test script.
- `forward()` returns **5 maps**: S0, S1, S2, S3, and the sum S1+S2+S3. `pred[-1]` is the sum.
- Training loss is `BCE + IoU` applied to **all five** outputs (deep supervision), matching
  Eq. 13 of the paper.
- All PVTv2 variants B1–B5 share `embed_dims = [64, 128, 320, 512]`; only `depths` differ. This
  is what makes the B2 → B4 swap a one-line change.
- `data_loader.py` computes a Canny edge map per sample and returns it as `data["edge"]`;
  `defect_train.py` never reads it.
- `colorEnhance()`, `randomGaussian()` and `randomPeper()` are defined in `data_loader.py` and
  never called.

## Evaluation protocol

Metrics come from the repo's own `sod_metrics.py`, the standard
[PySODMetrics](https://github.com/lartpang/PySODMetrics) implementation, so numbers are
computed identically to the paper.

Per image: resize to 384×384 → ImageNet normalisation → forward → take `pred[-1]` → sigmoid →
per-image min–max stretch → 8-bit greyscale → **resize back to the ground-truth resolution** →
score.

That last step matters: the score is computed at the image's native resolution, not at
384×384.

## Target numbers

| | MAE ↓ | wFβ ↑ | Sα ↑ | mFβ ↑ | mEξ ↑ |
|---|---|---|---|---|---|
| Paper, CrackSeg9k | .0135 | .7672 | .8493 | .7679 | .9481 |

Anything within roughly ±0.005 is normal reproduction noise from library versions and JPEG
decoding. **Whatever we measure becomes our baseline** — improvements are compared against
that, not against the printed values.
