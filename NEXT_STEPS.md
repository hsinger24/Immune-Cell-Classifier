# Immune Cell Classifier — Next Steps

The initial plan is complete: EfficientNetV2-S is trained on both the official and leakage-free grouped splits, checkpoints are saved in `models/`, and baseline results (confusion matrices, Grad-CAM, TTA) are in `reports/`. Headline numbers: macro-F1 = 1.0000 on the grouped split, 0.8914 with TTA on the official split.

The next phase is about **communication**, not capability: build a suite of polished, PI-ready visuals that make the model's behavior legible at a glance.

## Priority 1 — PI-Ready Visual Suite

The existing `gradcam_*.png` panels are a good start but are low-density (one example per class, small thumbnails, default colormap). The goal is a set of figures that each tell a clean, standalone story and could drop straight into a lab meeting slide or a manuscript figure.

### 1. Grad-CAM 2.0 — Upgraded attention panels

A single high-DPI figure per split with:

- 4 rows (one per class) × 6 columns (6 diverse, high-confidence examples per class)
- Each cell shows the original crop with a semi-transparent Grad-CAM overlay (jet or inferno colormap, 0.45 alpha) and the model's predicted probability annotated in the corner
- Shared colorbar on the right
- Use the last conv block of EfficientNetV2-S (already wired in `src/eval.py`) and upsample with bicubic interpolation, not nearest, so the heatmap looks smooth
- Save at 300 DPI as both PNG and PDF (`reports/figures/gradcam_panel_grouped.{png,pdf}`)

### 2. Grad-CAM++ and Score-CAM comparison

A 1×3 grid for a representative image from each class showing: original | Grad-CAM | Grad-CAM++ | Score-CAM. This demonstrates robustness of the attention and is a common ask from reviewers. Use `pytorch-grad-cam` (already compatible with torch 2.2) to avoid reimplementing the variants.

### 3. Misclassification gallery with attention

Pull the 12 most-confidently-wrong predictions from the official split (this is where the signal is — the grouped split is saturated). For each, show:

- Original image
- Grad-CAM for the true class
- Grad-CAM for the predicted (wrong) class
- A small bar chart of the top-3 predicted probabilities

This figure directly answers "where and why does the model fail" and is usually the most-discussed figure in a talk.

### 4. Mean attention per class

Aggregate Grad-CAM heatmaps across ~200 correctly-classified examples per class, aligned to the image center, and average. This gives a single "prototypical" heatmap per class showing which spatial regions the model relies on for eosinophils vs. lymphocytes vs. monocytes vs. neutrophils. Plot as a 1×4 panel. This is the kind of visual that gets circulated beyond the immediate lab.

### 5. t-SNE / UMAP embedding plot

Extract the penultimate-layer features (pre-classifier, 1280-dim for EfficientNetV2-S) for the entire grouped test set. Project with UMAP (preferred over t-SNE for preserving global structure) and scatter-plot colored by true class, with misclassified points outlined in black. Target: one 8×8 inch figure with a clean legend. This visually demonstrates class separability in feature space — a very strong "the model has learned something real" figure.

### 6. Reliability diagram + per-class ROC/PR curves

A 1×3 figure:

- Reliability diagram (predicted confidence vs. empirical accuracy) with ECE annotated
- Per-class ROC curves with AUC in the legend
- Per-class precision-recall curves with AP in the legend

If ECE > 0.05, add temperature-scaled calibration overlay.

### 7. Confusion matrix, upgraded

Replace the existing matrices with a single normalized-row version, annotated with both counts and percentages, using a diverging colormap centered on zero-error. Add marginal class-frequency bars on the axes. One figure, two variants (official, grouped), saved at 300 DPI.

### 8. Cross-dataset sanity check (stretch visual)

Run the model on the raw 640×480 smears in `dataset-master/` after center-cropping around each VOC bounding box. Plot a gallery of 16 predictions with bounding boxes, predicted label, and confidence — cells the model has never seen in any form. This is the single most persuasive visual for "does this actually generalize."

## Priority 2 — One-page summary figure

After the individual figures exist, assemble a single 2×3 "hero figure" that can stand alone in a slide:

- Top row: sample data panel | confusion matrix (grouped) | confusion matrix (official)
- Bottom row: Grad-CAM panel (compressed) | UMAP embedding | reliability diagram

Save as `reports/figures/hero_figure.{png,pdf}` at 300 DPI. Target dimensions 16×10 in.

## Priority 3 — Delivery

- Collect all figures under `reports/figures/` with a short caption file (`reports/figures/CAPTIONS.md`) containing a 1–2 sentence draft caption for each — the PI can reuse these verbatim.
- Regenerate `reports/results.md` to embed the new figures and retire the old low-res ones.
- Build a lightweight `reports/figures/index.html` gallery so the whole set can be reviewed in a browser without opening 10 files.
- Export a 6–8 slide `.pptx` summary (title, problem, data, method, headline result, Grad-CAM, error analysis, conclusion) using the new figures — suitable for the PI to present with minimal editing.

## Implementation Notes

- Create `src/visualize.py` as the single entrypoint that regenerates every figure from the saved checkpoints. Make it idempotent and cheap enough to re-run after any future retraining.
- Pin matplotlib style: `plt.rcParams['figure.dpi'] = 150; plt.rcParams['savefig.dpi'] = 300; plt.rcParams['font.family'] = 'DejaVu Sans'`. Use a consistent class color palette across every figure (e.g., tab10 indices 0–3 mapped to the four classes) so cross-figure comparisons are trivial.
- Use `pytorch-grad-cam`, `umap-learn`, and `scikit-learn`'s calibration utilities — all drop-in, no new ML required.
- All visuals should run on MPS or CPU in well under 10 minutes total; no new training needed.

## Deferred / Optional

These are the non-visual items that were left on the original plan. They are genuinely optional given that the grouped-split F1 is already 1.0:

- Stain normalization (Macenko) and higher-resolution (384×384) retraining
- Ensemble of 3 seeds or 3 architectures
- Self-supervised pretraining on `dataset-master`
- Temperature-scaled calibration if the reliability diagram in §6 shows it's needed
