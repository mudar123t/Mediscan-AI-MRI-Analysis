"""Classification model factory.

* ``simplecnn``: the semester-1 ``SimpleCNN`` (``mediscan.classification.cnn``),
  unchanged; it expects 1×256×256 input in [0, 1] (fair baseline).
* ``resnet18`` / ``efficientnet_b0``: timm models with ImageNet weights. timm's
  ``in_chans=1`` sums the pretrained RGB filters into one input channel; the
  input is standardised with the grey ImageNet mean/std (as in ``models.seg``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mediscan.classification.cnn import SimpleCNN

_IMAGENET_GRAY_MEAN = 0.449
_IMAGENET_GRAY_STD = 0.226


class TimmClassifier(nn.Module):
    def __init__(self, arch: str, num_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.net = timm.create_model(arch, pretrained=pretrained, in_chans=1, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - _IMAGENET_GRAY_MEAN) / _IMAGENET_GRAY_STD)


def build_cls_model(name: str, num_classes: int = 3, **kwargs: Any) -> nn.Module:
    """Build a classifier by name; returns logits of shape (B, num_classes)."""
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    if name in ("resnet18", "efficientnet_b0"):
        return TimmClassifier(name, num_classes=num_classes, pretrained=bool(kwargs.get("pretrained", True)))
    raise ValueError(f"unknown classification model {name!r}")
