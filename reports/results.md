# Immune Cell Classifier — Results

EfficientNetV2-S fine-tuned on the Kaggle BCCD-derived 4-class blood cell dataset.
Two evaluation regimes are reported: the **official** train/test split and a **grouped** split where augmented copies of the same source cell cannot straddle splits.

## Headline metrics

| Split    | Accuracy | Macro-F1 | ECE   |
|----------|---------:|---------:|------:|
| grouped  | 1.0000   | 1.0000   | 0.064 |
| official | 0.8838   | 0.8867   | 0.039 |

## Hero figure

![Hero figure](figures/hero_figure.png)

## Priority 1 — Visuals

### gradcam_panel_grouped.png

![gradcam_panel_grouped.png](figures/gradcam_panel_grouped.png)

Grad-CAM attention on 24 high-confidence correct predictions from the grouped test split. The model consistently localizes the nucleus/cytoplasm region characteristic of each leukocyte class.

### gradcam_panel_official.png

![gradcam_panel_official.png](figures/gradcam_panel_official.png)

Same panel layout applied to the official split, showing that attention remains nucleus-centered even where augmentation-induced leakage is absent.

### cam_comparison_grouped.png

![cam_comparison_grouped.png](figures/cam_comparison_grouped.png)

Grad-CAM, Grad-CAM++ and Score-CAM agree on the informative region for a representative high-confidence example of each class, confirming that the attention signal is robust to method choice.

### misclass_gallery_official.png

![misclass_gallery_official.png](figures/misclass_gallery_official.png)

The model's most-confidently-wrong predictions on the official test split, paired with attention maps for the true and predicted classes and a top-3 probability bar chart.

### mean_attention_grouped.png

![mean_attention_grouped.png](figures/mean_attention_grouped.png)

Grad-CAM heatmap averaged over up to 200 correctly classified examples per class, showing the prototypical spatial attention pattern the model uses for each leukocyte.

### umap_grouped.png

![umap_grouped.png](figures/umap_grouped.png)

UMAP projection of the 1280-D penultimate-layer features on the grouped test set. Clear class-wise clustering indicates the model has learned a meaningful embedding; misclassified points (outlined in black) sit at cluster boundaries.

### calibration_grouped.png

![calibration_grouped.png](figures/calibration_grouped.png)

Reliability diagram with ECE, one-vs-rest ROC and precision-recall curves on the grouped test set.

### calibration_official.png

![calibration_official.png](figures/calibration_official.png)

Same calibration panel on the official split; residual confusion between lymphocytes and monocytes shows up as lower AP and a small calibration gap.

### confusion_matrix_v2_grouped.png

![confusion_matrix_v2_grouped.png](figures/confusion_matrix_v2_grouped.png)

Row-normalized confusion matrix (grouped split) with raw counts, percentages and class-count marginals.

### confusion_matrix_v2_official.png

![confusion_matrix_v2_official.png](figures/confusion_matrix_v2_official.png)

Same, on the official split.

### cross_dataset_sanity.png

![cross_dataset_sanity.png](figures/cross_dataset_sanity.png)

Predictions on 16 raw blood smears from dataset-master — a completely separate VOC-style collection the model has never seen during training.

Browse all figures as a gallery: [figures/index.html](figures/index.html). Draft captions: [figures/CAPTIONS.md](figures/CAPTIONS.md).