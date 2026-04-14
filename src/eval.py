"""
Evaluation module for the blood cell classifier.

Features:
  - Full metric report: accuracy, macro-F1, per-class precision/recall
  - Confusion matrix visualization
  - Test-time augmentation (TTA)
  - Grad-CAM visualization
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from .data import (
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    get_grouped_split_loaders,
    get_official_split_loaders,
)
from .model import create_model


# ── Core evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run evaluation and return detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        _, preds = logits.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0)

    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "logits": all_logits,
    }


# ── Test-Time Augmentation ──────────────────────────────────────────────────

def _tta_transforms(img_size: int = 224) -> list[T.Compose]:
    """Return a list of deterministic augmentations for TTA."""
    base = [
        T.Resize(img_size + 32),
        T.CenterCrop(img_size),
    ]
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    variants = [
        base + norm,                                          # original
        base + [T.RandomHorizontalFlip(p=1.0)] + norm,       # h-flip
        base + [T.RandomVerticalFlip(p=1.0)] + norm,         # v-flip
        base + [T.RandomRotation((90, 90))] + norm,           # 90°
        base + [T.RandomRotation((180, 180))] + norm,         # 180°
        base + [T.RandomRotation((270, 270))] + norm,         # 270°
    ]
    return [T.Compose(v) for v in variants]


@torch.no_grad()
def evaluate_with_tta(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Evaluate with test-time augmentation (average logits over augmentations)."""
    from .data import _SubsetWithTransform, CellDataset, _CombinedDataset

    model.eval()
    tta_tfms = _tta_transforms(img_size)

    # Collect labels from the first pass
    all_logits_sum = None
    all_labels = None

    for i, tfm in enumerate(tta_tfms):
        # Re-wrap the dataset's base data with a new transform
        if hasattr(dataset, 'dataset'):
            # It's a _SubsetWithTransform
            tta_ds = _SubsetWithTransform(dataset.dataset, dataset.indices, tfm)
        else:
            # It's a direct dataset — apply transform
            dataset.transform = tfm
            tta_ds = dataset

        loader = DataLoader(tta_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        batch_logits = []
        batch_labels = []
        for images, labels in tqdm(loader, desc=f"  TTA {i+1}/{len(tta_tfms)}", leave=False):
            images = images.to(device)
            logits = model(images)
            batch_logits.append(logits.cpu())
            batch_labels.extend(labels.numpy())

        logits_tensor = torch.cat(batch_logits, dim=0)
        if all_logits_sum is None:
            all_logits_sum = logits_tensor
            all_labels = np.array(batch_labels)
        else:
            all_logits_sum += logits_tensor

    # Average logits
    avg_logits = all_logits_sum / len(tta_tfms)
    all_preds = avg_logits.argmax(dim=1).numpy()

    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "logits": avg_logits,
    }


# ── Confusion matrix plot ────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        ax=ax, square=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """Simple Grad-CAM for timm models."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def get_last_conv_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Get the last convolutional layer for Grad-CAM."""
    if "efficientnet" in model_name:
        # EfficientNetV2: last block in the features
        return model.conv_head
    elif "resnet" in model_name:
        return model.layer4[-1]
    elif "convnext" in model_name:
        return model.stages[-1].blocks[-1]
    raise ValueError(f"Unknown model: {model_name}")


def plot_gradcam_grid(
    model: torch.nn.Module,
    model_name: str,
    dataset,
    device: torch.device,
    n_per_class: int = 4,
    img_size: int = 224,
    save_path: str | Path | None = None,
) -> None:
    """Plot a grid of images with Grad-CAM overlays, organized by class."""
    import torchvision.transforms.functional as TF

    target_layer = get_last_conv_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    # Inverse normalization for display
    inv_normalize = T.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1.0 / s for s in IMAGENET_STD],
    )

    fig, axes = plt.subplots(NUM_CLASSES, n_per_class * 2, figsize=(n_per_class * 4, NUM_CLASSES * 2.5))

    # Collect indices per class
    class_indices = {c: [] for c in range(NUM_CLASSES)}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[label].append(i)
        if all(len(v) >= n_per_class for v in class_indices.values()):
            break

    for cls_idx in range(NUM_CLASSES):
        indices = class_indices[cls_idx][:n_per_class]
        for j, idx in enumerate(indices):
            img_tensor, label = dataset[idx]
            if not isinstance(img_tensor, torch.Tensor):
                # Need to apply transform
                tfm = T.Compose([
                    T.Resize(img_size + 32), T.CenterCrop(img_size),
                    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
                img_tensor = tfm(img_tensor)

            input_t = img_tensor.unsqueeze(0).to(device)

            # Get prediction and cam
            with torch.enable_grad():
                cam = grad_cam(input_t)

            pred = model(input_t).argmax(1).item()

            # Display original image
            display_img = inv_normalize(img_tensor).clamp(0, 1).permute(1, 2, 0).numpy()

            # Resize cam to image size
            import cv2
            cam_resized = cv2.resize(cam, (img_size, img_size)) if cam.shape != (img_size, img_size) else cam

            # Original
            ax = axes[cls_idx, j * 2]
            ax.imshow(display_img)
            ax.set_title(f"True: {CLASSES[label]}\nPred: {CLASSES[pred]}", fontsize=7)
            ax.axis("off")

            # Grad-CAM overlay
            ax2 = axes[cls_idx, j * 2 + 1]
            ax2.imshow(display_img)
            ax2.imshow(cam_resized, cmap="jet", alpha=0.4)
            ax2.set_title("Grad-CAM", fontsize=7)
            ax2.axis("off")

    plt.suptitle("Grad-CAM Visualizations by Class", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Grad-CAM grid to {save_path}")
    plt.close()


# ── Load and evaluate ────────────────────────────────────────────────────────

def load_and_evaluate(
    checkpoint_path: str | Path,
    split_mode: str = "official",
    use_tta: bool = False,
    save_dir: str | Path = "reports",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Load a checkpoint and run full evaluation."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_name = ckpt["model_name"]
    model = create_model(model_name, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} from {checkpoint_path}")
    print(f"  Trained epoch: {ckpt['epoch']}, phase: {ckpt['phase']}, val_f1: {ckpt['val_f1']:.4f}")

    # Data
    if split_mode == "official":
        _, _, test_loader = get_official_split_loaders(
            img_size=img_size, batch_size=batch_size, num_workers=num_workers)
    else:
        _, _, test_loader = get_grouped_split_loaders(
            img_size=img_size, batch_size=batch_size, num_workers=num_workers)

    # Evaluate
    if use_tta:
        results = evaluate_with_tta(model, test_loader.dataset, device,
                                     img_size=img_size, batch_size=batch_size,
                                     num_workers=num_workers)
    else:
        results = evaluate(model, test_loader, device)

    # Print results
    suffix = f"{'tta_' if use_tta else ''}{split_mode}"
    print(f"\n{'='*60}")
    print(f"Results ({suffix})")
    print(f"{'='*60}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Macro-F1:  {results['macro_f1']:.4f}")
    print(f"\n{results['report']}")

    # Save confusion matrix
    cm_path = save_dir / f"confusion_matrix_{suffix}.png"
    plot_confusion_matrix(results["confusion_matrix"],
                          title=f"Confusion Matrix ({suffix})", save_path=cm_path)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate blood cell classifier")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="official",
                        choices=["official", "grouped"])
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualizations")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    results = load_and_evaluate(
        args.checkpoint,
        split_mode=args.split,
        use_tta=args.tta,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
