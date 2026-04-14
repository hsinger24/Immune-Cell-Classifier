"""
Generate the final results report after training.

Usage:
    python -m src.generate_report
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np

from .data import (
    CLASSES,
    get_grouped_split_loaders,
    get_official_split_loaders,
)
from .eval import (
    evaluate,
    evaluate_with_tta,
    get_last_conv_layer,
    load_and_evaluate,
    plot_confusion_matrix,
    plot_gradcam_grid,
)
from .model import create_model


def generate_report(
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
):
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    report_lines = [
        "# Blood Cell Classifier — Results Report\n",
        "## Model: EfficientNetV2-S\n",
        "Two evaluation modes are reported:\n",
        "1. **Official split** — uses dataset2's provided TRAIN/TEST split (contains data leakage)",
        "2. **Grouped split** — group-aware split by sourceID (leakage-free, true generalization)\n",
    ]

    for split_mode in ["official", "grouped"]:
        ckpt_path = models_dir / f"best_efficientnetv2_s_{split_mode}.pt"
        if not ckpt_path.exists():
            report_lines.append(f"\n### {split_mode.title()} Split\n")
            report_lines.append(f"*Checkpoint not found at {ckpt_path}*\n")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = create_model(ckpt["model_name"], pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        model.eval()

        # Get test loader
        if split_mode == "official":
            _, _, test_loader = get_official_split_loaders(
                img_size=img_size, batch_size=batch_size, num_workers=num_workers)
        else:
            _, _, test_loader = get_grouped_split_loaders(
                img_size=img_size, batch_size=batch_size, num_workers=num_workers)

        # Standard evaluation
        results = evaluate(model, test_loader, device)

        # TTA evaluation
        results_tta = evaluate_with_tta(
            model, test_loader.dataset, device,
            img_size=img_size, batch_size=batch_size, num_workers=num_workers)

        # Confusion matrices
        plot_confusion_matrix(
            results["confusion_matrix"],
            title=f"Confusion Matrix — {split_mode.title()} Split",
            save_path=reports_dir / f"confusion_matrix_{split_mode}.png")
        plot_confusion_matrix(
            results_tta["confusion_matrix"],
            title=f"Confusion Matrix — {split_mode.title()} Split (TTA)",
            save_path=reports_dir / f"confusion_matrix_{split_mode}_tta.png")

        # Grad-CAM
        try:
            plot_gradcam_grid(
                model, ckpt["model_name"], test_loader.dataset, device,
                n_per_class=3, img_size=img_size,
                save_path=reports_dir / f"gradcam_{split_mode}.png")
        except Exception as e:
            print(f"Grad-CAM failed for {split_mode}: {e}")

        # Write to report
        report_lines.append(f"\n### {split_mode.title()} Split\n")
        report_lines.append(f"- **Checkpoint:** epoch {ckpt['epoch']}, phase: {ckpt['phase']}")
        report_lines.append(f"- **Val macro-F1 (during training):** {ckpt['val_f1']:.4f}\n")

        report_lines.append("#### Standard Evaluation\n")
        report_lines.append(f"- **Accuracy:** {results['accuracy']:.4f}")
        report_lines.append(f"- **Macro-F1:** {results['macro_f1']:.4f}\n")
        report_lines.append("```")
        report_lines.append(results["report"])
        report_lines.append("```\n")

        report_lines.append("#### With Test-Time Augmentation (TTA)\n")
        report_lines.append(f"- **Accuracy:** {results_tta['accuracy']:.4f}")
        report_lines.append(f"- **Macro-F1:** {results_tta['macro_f1']:.4f}\n")
        report_lines.append("```")
        report_lines.append(results_tta["report"])
        report_lines.append("```\n")

        report_lines.append(f"![Confusion Matrix]({f'confusion_matrix_{split_mode}.png'})\n")
        report_lines.append(f"![Confusion Matrix TTA]({f'confusion_matrix_{split_mode}_tta.png'})\n")
        report_lines.append(f"![Grad-CAM]({f'gradcam_{split_mode}.png'})\n")

    # Leakage discussion
    report_lines.extend([
        "\n## Data Leakage Discussion\n",
        "The official dataset2 TRAIN/TEST split has **100% sourceID overlap** — every TEST image is an",
        "augmentation of a cell that also appears (in different augmented forms) in TRAIN. This means the",
        "official split results overestimate generalization. The model partially memorizes individual cell",
        "identity rather than purely learning cell-type morphology.\n",
        "The grouped split separates by sourceID, ensuring no cell appears in more than one split.",
        "This gives a more honest measure of how the model would perform on truly unseen cells.\n",
        "Both numbers are reported for completeness — the official split for comparison with published",
        "baselines, and the grouped split as our trusted measure of real-world performance.\n",
    ])

    # Success criteria
    report_lines.extend([
        "\n## Success Criteria Assessment\n",
        "| Criterion | Target | Result | Status |",
        "|---|---|---|---|",
    ])

    # Write report
    report_path = reports_dir / "results.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    generate_report()
