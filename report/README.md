# Report — build instructions

`main.tex` is written for the official **CVPR** LaTeX template. It does not include
`cvpr.sty`, because that file belongs to the conference and should be taken from the
official template rather than copied around.

## Compiling on Overleaf (recommended)

1. Go to [overleaf.com](https://www.overleaf.com) → **New Project** → **Templates** →
   search **"CVPR 2025"** (or the most recent year available) → **Open as Template**.
2. In the new project, delete the template's `main.tex`.
3. Upload from this folder:
   - `main.tex`
   - `refs.bib`
   - the whole `figures/` folder
4. Set the compiler to **pdfLaTeX** (Menu → Compiler).
5. Press **Recompile**. Overleaf runs BibTeX automatically; if citations show as `[?]`,
   recompile a second time.

The template already provides `cvpr.sty` and `ieeenat_fullname.bst`. **Do not edit them.**

## Switching from review to camera-ready

`main.tex` line 8 currently reads:

```latex
\usepackage[review]{cvpr}
```

This adds line numbers and the review banner. For the final submission change it to:

```latex
\usepackage{cvpr}
```

and fill in Darrsheni's roll number on line 28 (currently `27PGAI00XX`).

## Files

| File | Purpose |
|---|---|
| `main.tex` | the paper |
| `refs.bib` | 14 references, all cited |
| `figures/architecture.png` | Fig. 1 — method overview |
| `figures/curves.png` | Fig. 2 — training and validation curves |
| `figures/qualitative_compare.png` | Fig. 3 — qualitative comparison |

## Length

Roughly 8 pages of body text plus references, which is the target. If it overruns after
compiling, the safest trims are §6.4 (practical implications) and the second paragraph of
§6.2 (cost analysis) — both are commentary rather than evidence.

If it runs **short**, the honest way to extend it is more analysis of the existing runs
(for example per-class or per-image error breakdowns from the saved prediction maps),
not more prose.

## A note on the numbers

Every figure in the paper is measured, and each traces to a `summary.json` in the results
directory produced by `wpformer_plus.py`. The published WPFormer row in Table 1 is quoted
from the original paper and is labelled as such. Nothing is estimated or projected.
