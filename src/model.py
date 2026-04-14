"""
Model factory for blood cell classification.

Supports:
  - EfficientNetV2-S  (primary)
  - ResNet-50
  - ConvNeXt-Tiny

All models are loaded with ImageNet-pretrained weights from timm.
"""

from __future__ import annotations

from typing import Literal

import timm
import torch
import torch.nn as nn

from .data import NUM_CLASSES

ModelName = Literal["efficientnetv2_s", "resnet50", "convnext_tiny"]

# Map our short names to timm model identifiers
_TIMM_NAMES = {
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "resnet50": "resnet50.a1_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
}


def create_model(
    name: ModelName = "efficientnetv2_s",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    drop_rate: float = 0.3,
) -> nn.Module:
    """Create a classification model with a fresh head.

    Args:
        name: One of 'efficientnetv2_s', 'resnet50', 'convnext_tiny'.
        num_classes: Number of output classes.
        pretrained: Load ImageNet weights.
        drop_rate: Dropout before the final classifier.

    Returns:
        A timm model with the classifier replaced.
    """
    timm_name = _TIMM_NAMES[name]
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head.

    Uses timm's get_classifier() to identify head parameters reliably
    across architectures (EfficientNet, ResNet, ConvNeXt, etc.).
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    # Then unfreeze the classifier head
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True
    # Also unfreeze head norm if present (ConvNeXt head.norm)
    if hasattr(model, "head") and hasattr(model.head, "norm"):
        for param in model.head.norm.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    for name in _TIMM_NAMES:
        print(f"\n=== {name} ===")
        m = create_model(name, pretrained=False)
        print(f"  Params: {count_parameters(m)}")

        freeze_backbone(m)
        print(f"  After freeze: {count_parameters(m)}")

        unfreeze_all(m)
        print(f"  After unfreeze: {count_parameters(m)}")

        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        print(f"  Output shape: {out.shape}")
