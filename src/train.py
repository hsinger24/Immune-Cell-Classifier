"""
Training loop for the blood cell classifier.

Features:
  - Two-phase training: head-only warmup → full fine-tuning
  - AdamW optimizer with cosine-annealing LR + linear warmup
  - Mixup augmentation
  - Label smoothing
  - Best-model checkpointing on macro-F1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    CLASSES,
    NUM_CLASSES,
    get_grouped_split_loaders,
    get_official_split_loaders,
)
from .model import (
    ModelName,
    count_parameters,
    create_model,
    freeze_backbone,
    unfreeze_all,
)

# ── Mixup ────────────────────────────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply mixup to a batch. Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Training step ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            # Accuracy approximation for mixup
            _, preds = outputs.max(1)
            correct += (lam * preds.eq(targets_a).sum().item()
                        + (1 - lam) * preds.eq(targets_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()

        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "lr": scheduler.get_last_lr()[0],
    }


# ── Validation step ──────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss": total_loss / len(all_labels),
        "acc": (all_preds == all_labels).mean(),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
    }


# ── Full training run ────────────────────────────────────────────────────────

def train(
    model_name: ModelName = "efficientnetv2_s",
    split_mode: str = "official",
    img_size: int = 224,
    batch_size: int = 32,
    head_epochs: int = 3,
    finetune_epochs: int = 25,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    save_dir: str | Path = "models",
    num_workers: int = 4,
) -> Path:
    """Run the full two-phase training pipeline.

    Returns the path to the best checkpoint.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_{model_name}_{split_mode}.pt"
    ckpt_path = save_dir / ckpt_name

    # ── Device ───────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    if split_mode == "official":
        train_loader, val_loader, test_loader = get_official_split_loaders(
            img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        )
    else:
        train_loader, val_loader, test_loader = get_grouped_split_loaders(
            img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        )

    # ── Model ────────────────────────────────────────────────────────────
    model = create_model(model_name, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ── Phase 1: Train head only ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: Training head only ({head_epochs} epochs)")
    print(f"{'='*60}")
    freeze_backbone(model)
    params = count_parameters(model)
    print(f"Trainable params: {params['trainable']:,} / {params['total']:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head, weight_decay=weight_decay,
    )
    total_steps = head_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_head, total_steps=total_steps,
    )

    best_f1 = 0.0

    for epoch in range(1, head_epochs + 1):
        t_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            use_mixup=False,  # no mixup during head warmup
        )
        v_metrics = validate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch}/{head_epochs} — "
              f"train_loss: {t_metrics['loss']:.4f}, train_acc: {t_metrics['acc']:.4f} | "
              f"val_loss: {v_metrics['loss']:.4f}, val_acc: {v_metrics['acc']:.4f}, "
              f"val_f1: {v_metrics['macro_f1']:.4f}")

        if v_metrics["macro_f1"] > best_f1:
            best_f1 = v_metrics["macro_f1"]
            torch.save({
                "model_state": model.state_dict(),
                "model_name": model_name,
                "epoch": epoch,
                "phase": "head",
                "val_f1": best_f1,
                "split_mode": split_mode,
            }, ckpt_path)

    # ── Phase 2: Fine-tune full network ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: Full fine-tuning ({finetune_epochs} epochs)")
    print(f"{'='*60}")
    unfreeze_all(model)
    params = count_parameters(model)
    print(f"Trainable params: {params['trainable']:,} / {params['total']:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_finetune, weight_decay=weight_decay,
    )
    warmup_steps = 3 * len(train_loader)
    total_steps = finetune_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6,
    )
    # Wrap with warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_steps],
    )

    patience = 7
    patience_counter = 0

    for epoch in range(1, finetune_epochs + 1):
        t_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            use_mixup=True, mixup_alpha=mixup_alpha,
        )
        v_metrics = validate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch}/{finetune_epochs} — "
              f"train_loss: {t_metrics['loss']:.4f}, train_acc: {t_metrics['acc']:.4f}, "
              f"lr: {t_metrics['lr']:.2e} | "
              f"val_loss: {v_metrics['loss']:.4f}, val_acc: {v_metrics['acc']:.4f}, "
              f"val_f1: {v_metrics['macro_f1']:.4f}")

        if v_metrics["macro_f1"] > best_f1:
            best_f1 = v_metrics["macro_f1"]
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_name": model_name,
                "epoch": head_epochs + epoch,
                "phase": "finetune",
                "val_f1": best_f1,
                "split_mode": split_mode,
            }, ckpt_path)
            print(f"  ↑ New best macro-F1: {best_f1:.4f} — saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement.")
                break

    print(f"\nTraining complete. Best val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    return ckpt_path


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train blood cell classifier")
    parser.add_argument("--model", type=str, default="efficientnetv2_s",
                        choices=["efficientnetv2_s", "resnet50", "convnext_tiny"])
    parser.add_argument("--split", type=str, default="official",
                        choices=["official", "grouped"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    train(
        model_name=args.model,
        split_mode=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        num_workers=args.num_workers,
    )
