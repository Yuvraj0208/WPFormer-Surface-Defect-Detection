# Report — build instructions

`main.tex` targets the **official CVPR 2026 author kit**
([cvpr-org/author-kit](https://github.com/cvpr-org/author-kit)). It deliberately does
**not** ship `cvpr.sty` or `ieeenat_fullname.bst` — those belong to the conference and
should come from the official kit rather than be copied around.

## Compiling on Overleaf

1. Open the [CVPR 2026 Submission Template](https://www.overleaf.com/latex/templates/cvpr-2026-submission-template/rdtrwgypxxzb)
   → **Open as Template**.
2. In the new project, **replace** these files with the ones from this folder:
   - `main.tex`
   - `main.bib` (overwrites the template's placeholder bibliography)
3. Upload the `figures/` folder with all three PNGs.
4. Compiler: **pdfLaTeX** (Menu → Compiler).
5. **Recompile twice.** BibTeX needs the second pass or citations render as `[?]`.

The template's own `preamble.tex`, `sec/` and `fig/` folders are left unused — `main.tex`
here is self-contained and loads everything it needs. You can delete them or leave them;
neither affects the build.

## Paper type

Line 10 selects the mode:

```latex
\usepackage{cvpr}                % CAMERA-READY  <- currently active
% \usepackage[review]{cvpr}      % line numbers + anonymised authors
% \usepackage[pagenumbers]{cvpr} % camera-ready look WITH page numbers
```

It is set to **camera-ready**, which is what "IEEE CVPR format" means and what you submit.

If your professor wants numbered pages for marking, switch to `[pagenumbers]` — that keeps
the camera-ready layout and just adds page numbers. Do **not** submit the `[review]`
version: it anonymises the author block, so your names disappear.

## Files

| File | Purpose |
|---|---|
| `main.tex` | the paper |
| `main.bib` | 14 references, all cited, none unused |
| `figures/architecture.png` | Fig. 1 — method overview |
| `figures/curves.png` | Fig. 2 — training and validation curves |
| `figures/qualitative_compare.png` | Fig. 3 — qualitative comparison |

## Length

Roughly 8 pages of body text plus references. **Check the page count after your first
compile** — float placement moves things around.

If it overruns, trim in this order: §6.4 (practical implications), then the second
paragraph of §6.2 (cost analysis). Both are commentary rather than evidence.

If it runs short, extend with more analysis of the runs already completed — for example
per-image error breakdowns from the saved prediction maps — rather than more prose.

## A note on the numbers

Every figure in the paper is measured, and each traces to a `summary.json` produced by
`wpformer_plus.py`. The published WPFormer row in Table 1 is quoted from the original
paper and labelled as such. Nothing is estimated, projected or rounded in our favour.
