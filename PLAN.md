# Immune Cell Classifier — Plan

## Progress

### Done
- [x] Project structure created (`src/`, `notebooks/`, `models/`, `reports/`)
- [x] Python 3.12 venv with all dependencies (PyTorch 2.2, timm, scikit-learn, etc.)
- [x] `src/data.py` — CellDataset, sourceID extraction, group-aware splits (GroupShuffleSplit), transforms, dataloaders
- [x] `src/model.py` — model factory (EfficientNetV2-S, ResNet-50, ConvNeXt-Tiny), freeze/unfreeze utilities
- [x] `src/train.py` — two-phase training (head-only → full fine-tune), AdamW, cosine LR + warmup, mixup, label smoothing, early stopping
- [x] `src/eval.py` — evaluation metrics, confusion matrix plotting, TTA, Grad-CAM
- [x] `notebooks/01_eda.ipynb` — EDA complete; confirmed 100% sourceID leakage, class balance, 320x240 images
- [x] `src/generate_report.py` — report generation script (ready to run after training)
- [x] Training started for EfficientNetV2-S on both splits (MPS/Apple Silicon)
  - Official split: val F1 reached **1.0000** by epoch 4 (confirms leakage inflation)
  - Grouped split: val F1 reached **0.9978** by epoch 5 (exceeds stretch goal of 0.90)

### Remaining
- [ ] Let training runs finish (early stopping or 25 epochs) and save final checkpoints
- [ ] Run `python -m src.generate_report` to produce `reports/results.md` with test-set metrics, confusion matrices, and Grad-CAM
- [ ] Phase 3 iteration (TTA, ensembling, higher-res input, stain normalization) — optional given current results already exceed all targets
- [ ] Phase 4 error analysis — inspect misclassifications, calibration check

### Setup on a New Machine

```bash
cd /path/to/Immune-Cell-Classifier

# 1. Create venv — must use Python 3.12 (PyTorch does not have wheels for 3.13 yet)
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Resume training (will auto-detect CUDA, MPS, or CPU)
python -m src.train --model efficientnetv2_s --split official --num-workers 4
python -m src.train --model efficientnetv2_s --split grouped --num-workers 4

# 3. Generate final report (after training finishes)
python -m src.generate_report
```

**Notes:**
- Existing partial checkpoints are in `models/` (val F1: 1.00 official, 0.9978 grouped). Training will overwrite them only if it finds a better score.
- The two training runs can be run in parallel if GPU memory allows, or sequentially.
- Adjust `--num-workers` based on available CPU cores.
- `--batch-size` defaults to 32; increase to 64 if GPU memory permits for faster training.

---

## Objective

Build the best possible classifier that distinguishes four white blood cell classes from microscopy images:

- **EOSINOPHIL**
- **LYMPHOCYTE**
- **MONOCYTE**
- **NEUTROPHIL**

Primary metric: **macro-averaged F1** on a held-out test set (weights all four classes equally, resistant to class imbalance). Secondary metrics: per-class precision/recall, confusion matrix, top-1 accuracy.

## Data Inventory

Two datasets are available in this workspace. They come from the same source project (Shenggan's BCCD / blood-cell-image dataset) but are organized differently.

### `dataset2-master/` — preprocessed single-cell crops (primary training data)

Pre-split, pre-cropped, augmented single-cell images at **320×240 RGB**. Approximately balanced across the four target classes.

| Split | EOSINOPHIL | LYMPHOCYTE | MONOCYTE | NEUTROPHIL | Total |
|---|---|---|---|---|---|
| TRAIN | 2,497 | 2,483 | 2,478 | 2,499 | 9,957 |
| TEST | 623 | 620 | 620 | 624 | 2,487 |
| TEST_SIMPLE | 13 | 6 | 4 | 48 | 71 |

Filename pattern `_<sourceID>_<augIndex>.jpeg` reveals these are **augmentation expansions of a much smaller set of original images** (~88 unique source IDs). This has a critical implication we must handle — see "Data leakage risk" below.

### `dataset-master/` — raw full-frame blood smears (secondary / validation)

410 full 640×480 blood smear images (`JPEGImages/`), each with a Pascal VOC XML bounding-box annotation (`Annotations/`) and an image-level class in `labels.csv`. Classes in the CSV include NEUTROPHIL, EOSINOPHIL, LYMPHOCYTE, MONOCYTE, BASOPHIL, and some multi-label rows (e.g., `"NEUTROPHIL, EOSINOPHIL"`). These are the upstream images from which `dataset2` was cropped.

## Critical Risk: Data Leakage in the Provided Split

Because `dataset2`'s TRAIN and TEST were produced by augmenting the same pool of ~88 original images, crops of the same underlying cell likely appear on both sides of the split. A model trained naively on TRAIN and evaluated on TEST will report **inflated accuracy** — it is partly memorizing cell identity rather than learning cell-type morphology.

**Mitigation:** Group by `sourceID` (the first number in the filename) and construct a **GroupKFold** or group-aware train/val/test split so that no source image appears in more than one split. We will report both:

1. **Official split** (TRAIN → TEST as provided) — comparable to published baselines.
2. **Leakage-free split** (group by sourceID) — our real measure of generalization. This is the number we trust.

## Approach

### Phase 1 — Baseline & EDA

1. Load both datasets; verify class balance, image sizes, and sourceID groupings.
2. Visualize 8–16 images per class; confirm staining, zoom, and framing are consistent.
3. Train a quick logistic-regression baseline on a pretrained ImageNet feature extractor (ResNet-50 penultimate layer, frozen) to establish a floor.
4. Target: >85% accuracy on the official split with the frozen-feature baseline.

### Phase 2 — Fine-tuned CNN (primary model)

1. **Architecture:** start with **EfficientNetV2-S** pretrained on ImageNet. Good accuracy/compute tradeoff and well-studied on microscopy. Also try ResNet-50 and ConvNeXt-Tiny as comparison points.
2. **Input:** resize to 224×224 (model's native resolution); preserve aspect ratio via center-crop after shortest-side resize.
3. **Augmentation (train only):** random horizontal + vertical flips, random 90° rotations, slight color jitter on hue/saturation (stain variation), small affine/zoom, Gaussian blur. Avoid aggressive color distortion — stain color carries diagnostic signal.
4. **Training:** AdamW, cosine LR schedule with 3-epoch warmup, label smoothing 0.1, mixup (α=0.2). Train head-only for 3 epochs, then unfreeze and fine-tune the full network for 20–30 epochs. Batch size 32–64.
5. **Loss:** cross-entropy with class weights (classes are near-balanced, so weights will be close to uniform).
6. **Model selection:** best macro-F1 on a grouped validation split carved from TRAIN.

### Phase 3 — Iteration for best performance

Explore these in order of expected payoff:

1. **Test-time augmentation (TTA):** average logits over flips + rotations at inference. Usually +1–2% accuracy for free.
2. **Ensembling:** average 3–5 models trained with different seeds/architectures.
3. **Higher-resolution input:** 320×320 or 384×384 to preserve nucleus detail — lymphocytes vs. monocytes can differ in fine nuclear texture.
4. **Stain normalization:** Macenko or Reinhard normalization to reduce stain-batch variation; useful if cross-dataset generalization matters.
5. **Self-supervised pretraining:** if time permits, SimCLR/DINO pretraining on `dataset-master` raw smears before fine-tuning.
6. **Use dataset-master bounding boxes:** re-crop single cells from raw smears using the VOC annotations to create an additional, non-augmented training pool. This also generates a genuinely held-out test set from images the model has never seen.

### Phase 4 — Evaluation & error analysis

1. Report on both the official split and the grouped split: accuracy, macro-F1, per-class precision/recall, confusion matrix.
2. Inspect misclassifications visually — especially eosinophil↔neutrophil confusion (both granulocytes) and lymphocyte↔monocyte (both mononuclear). These are the clinically meaningful error modes.
3. Calibration check (reliability diagram); apply temperature scaling if needed.
4. Grad-CAM on correct and incorrect predictions to sanity-check that the model attends to the nucleus/cytoplasm and not to staining background or image borders.

## Deliverables

1. `PLAN.md` — this document.
2. `notebooks/01_eda.ipynb` — data profiling and visualization.
3. `src/data.py`, `src/model.py`, `src/train.py`, `src/eval.py` — reusable training code.
4. `models/best.pt` — final checkpoint.
5. `reports/results.md` — metrics on both splits, confusion matrices, Grad-CAM samples, and a short discussion of the leakage caveat.

## Success Criteria

- **Must:** macro-F1 ≥ 0.92 on the official `dataset2` TEST split.
- **Should:** macro-F1 ≥ 0.85 on the leakage-free grouped split (the real measure).
- **Stretch:** macro-F1 ≥ 0.90 on the grouped split, and correct classification of a majority of the raw `dataset-master` smears after single-cell cropping.

## Open Questions

1. Are we optimizing for a research benchmark number (use the official split) or for real-world deployment (use the grouped split)? Both will be reported, but this changes which number we headline.
2. Do we need to also handle BASOPHIL or multi-label smears from `dataset-master`, or is 4-class single-label sufficient? Current plan assumes the latter.
3. Compute budget — GPU available? If CPU-only, we'll lean more heavily on frozen-feature baselines and smaller backbones (MobileNetV3, EfficientNet-B0).
