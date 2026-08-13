# WPFormer explained in plain language

A walkthrough of *Wavelet and Prototype Augmented Query-based Transformer for Pixel-level
Surface Defect Detection* (Yan et al., CVPR 2025, pp. 23860–23869) written for someone who
has not read the paper.

---

## 1. The task

A factory makes steel sheets; sometimes one has a scratch. A bridge develops a crack. A roll
of fabric has a flaw. Someone has to find these, and doing it by eye is slow, expensive and
occasionally dangerous.

We do not just want the computer to say *"this photo contains a crack"*. We want it to colour
in **exactly which pixels** are crack. The output is a black-and-white image the same size as
the input: white = defect, black = normal. That is **pixel-level segmentation**.

Three properties make industrial defects harder than everyday objects:

- **Thin.** A hairline crack may be three pixels wide in a 384×384 image.
- **Low contrast.** A grey crack on grey concrete barely differs from its surroundings.
- **Rare.** Roughly 1.4% of pixels are defect. Guessing "no defect" everywhere scores 98.6%.

## 2. What everyone did before, and why it is weak

The standard recipe: a network reads the image and produces feature maps; at the very end a
**convolution layer** slides across every pixel asking "defect or not?".

The paper's objection is that this final layer is **identical for every image**. Once training
ends its numbers are frozen — one rubber stamp pressed onto every pixel of every photo, for
ever. The paper calls this a single *static, image-independent query* that "lacks semantic
representation".

Bright scratch on polished steel? Same stamp. Faint crack on rough concrete? Same stamp. This
is why such models miss weak defects and get fooled by cluttered backgrounds.

## 3. Queries

Borrowed from DETR → MaskFormer → Mask2Former. Instead of one frozen stamp, use **16 learnable
queries**, each a 64-number vector. Think of them as detectives:

1. Each **looks at this specific image** and updates itself from what it sees — this is the
   crucial difference; they become image-specific.
2. Each produces **its own candidate mask**.
3. Each produces a **confidence weight** — how much to trust it *on this image*.
4. The final mask is the weighted blend of all 16 opinions.

The remaining question is *how* the queries look at the image. That is where the paper's two
contributions live.

## 4. WCA — Wavelet-enhanced Cross-Attention

### The intuition

Any image splits into two kinds of content:

- **Low frequency** — the big blurry shapes. Squint, and what remains is low frequency.
- **High frequency** — sharp edges and fine texture.

In music terms: bass and treble.

A crack is a thin sharp line, so it lives almost entirely in the **high-frequency** part.
Deliberately isolate that part and faint cracks become visible.

### The split

The **Haar wavelet transform** — an old, classical, non-learned operation — splits a feature
map into four half-size sub-bands:

| Sub-band | Content |
|---|---|
| `LL` | low frequency — overall structure |
| `LH` | high frequency, horizontal edges |
| `HL` | high frequency, vertical edges |
| `HH` | high frequency, diagonal edges |

The three high bands are summed into one "all the sharp detail" map.

### The catch, and the fix

High frequency carries crack edges **and** every speck of sensor noise and surface grain.
Turning up the treble gives you crisp cymbals and tape hiss together.

So the module learns a volume knob. It feeds high + low into a small context module that
predicts two sets of weights:

- **global** — one per channel, for the whole image ("channel 7 is mostly noise, turn it down")
- **local** — one per channel *per location* ("noise in this corner, but real along that line")

These are added, squashed by a sigmoid into a 0–1 multiplier, and multiplied onto the
high-frequency map. Noise down, real edges intact. The cleaned band is recombined with the low
band, and the queries read from **that**.

> WCA lets the queries look at a version of the image where thin edges are amplified and noise
> is suppressed.

## 5. PCA — Prototype-guided Cross-Attention

### The problem

At 1/8 resolution a feature map has roughly 2,300 positions, nearly all background. Comparing
each query against every position lets background **dilute** the attention that should go to
the handful of defect pixels.

### Why the obvious fix is wrong

Mask2Former and PEM use a **mask prior**: guess where the defect is, then let queries attend
only inside the guess.

The paper's objection: **a bad guess is unrecoverable.** If the prior misses half the crack,
later layers are forbidden from looking there, and that half is lost for good. Errors
propagate rather than getting corrected.

### Summarise instead of mask

Boil the ~2,300 positions down to **16 prototypes**. Two small convolutions produce 16 score
maps; a softmax turns them into "how strongly does each pixel belong to prototype *k*"; a
matrix multiply yields the prototypes. One ends up representing crack-like content, another
smooth background, another shadow, and so on.

Crucially, **nothing is discarded** — every pixel still contributes somewhere. That is the
difference from masking, where excluded pixels simply vanish.

Queries then interact with 16 summaries instead of 2,300 noisy positions, using the same
global + local weighting idea as WCA.

Two differences from PEM, stated in the paper: PEM builds prototypes by masked cross-attention
whereas PCA uses adaptive clustering; and PEM captures only local query–prototype
relationships whereas PCA captures both global and local.

## 6. Putting it together

```
Image 384x384
  -> PVTv2-B2 backbone      -> 4 maps at 1/4, 1/8, 1/16, 1/32  (64,128,320,512 ch)
  -> 1x1 convs              -> all squeezed to 64 channels
  -> FPN                    -> F1 (1/4, high-res) + F2,F3,F4
  -> 16 queries warmed up on F1 by a 2-layer transformer
  -> 3 x D2T decoder blocks, each = WCA -> PCA -> self-attention
  -> SegHead after each block -> S0, S1, S2, S3
  -> final prediction = S1 + S2 + S3
```

**Self-attention** at the end of each block lets the queries talk to each other, so they do
not all redundantly find the same thing.

**The segmentation head** passes each query through a 3-layer MLP, multiplies it against the
high-resolution map F1 to get that query's mask, and blends the 16 masks using per-query
weights from a linear layer.

## 7. Training

Five outputs, all graded — *deep supervision*, like a teacher marking every step of the
working rather than only the final answer:

```
L_total = sum over i in {0,1,2,3} of L(S_i, G)  +  L(S1+S2+S3, G)
L       = BCE + IoU loss
```

BCE alone would be dominated by the 98.6% background; the IoU term measures overlap of the
crack region itself and cannot be fooled the same way.

Settings for CrackSeg9k: Adam, lr 8e-5, cosine decay, batch 4, 60 epochs, one RTX 3090,
input resized to 384×384.

## 8. Results

Compared against 17 methods on three datasets, best on all. CrackSeg9k:

| Method | MAE ↓ | wFβ ↑ | Sα ↑ | mFβ ↑ | mEξ ↑ |
|---|---|---|---|---|---|
| Mask2Former | .0147 | .7442 | .8385 | .7478 | .9363 |
| PEM (CVPR'24) | .0146 | .7414 | .8333 | .7452 | .9354 |
| **WPFormer** | **.0135** | **.7672** | **.8493** | **.7679** | **.9481** |

### What the metrics mean

- **MAE** — average per-pixel error. Lower is better.
- **wFβ** — balances "did you find the cracks" against "did you cry wolf", weighting errors
  near the crack more heavily. The headline number.
- **Sα** — whether the predicted shape *structurally* resembles the true one.
- **mFβ / mEξ** — the same ideas averaged over all binarisation thresholds.

The ablation shows WCA alone reaches .7583 and PCA alone .7579, but **both together reach
.7672** — the two ideas help in different ways and compound. Sixteen queries beat both 8
and 64.
