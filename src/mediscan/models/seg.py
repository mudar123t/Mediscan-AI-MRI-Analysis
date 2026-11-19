"""Segmentation model factory.

* ``unet_s1``: the semester-1 U-Net (``mediscan.segmentation.unet.UNet``), kept
  unchanged as the fair baseline architecture.
* ``unet_r34``: U-Net with an ImageNet-pretrained ResNet-34 encoder
  (segmentation-models-pytorch), with ``Dropout2d`` after every decoder block.
  The dropout layers stay active at test time for MC-dropout sampling (M1.2).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mediscan.segmentation.unet import UNet

# Single-channel versions of the ImageNet mean/std (channel average).
_IMAGENET_GRAY_MEAN = 0.449
_IMAGENET_GRAY_STD = 0.226


class _DropoutAfter(nn.Module):
    """Wrap a module and apply ``Dropout2d`` to its output (keeps any call signature)."""

    def __init__(self, block: nn.Module, p: float) -> None:
        super().__init__()
        self.block = block
        self.drop = nn.Dropout2d(p)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.drop(self.block(*args, **kwargs))


class PretrainedUNet(nn.Module):
    """smp U-Net with input standardisation and decoder dropout."""

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,  # smp sums the pretrained RGB filters into one channel
            classes=1,
        )
        if dropout > 0:
            self.net.decoder.blocks = nn.ModuleList(_DropoutAfter(b, dropout) for b in self.net.decoder.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - _IMAGENET_GRAY_MEAN) / _IMAGENET_GRAY_STD)


def build_seg_model(name: str, **kwargs: Any) -> nn.Module:
    """Build a segmentation model by name (returns logits of shape (B, 1, H, W))."""
    if name == "unet_s1":
        return UNet(in_channels=1, out_channels=1)
    if name == "unet_r34":
        return PretrainedUNet(encoder="resnet34", **kwargs)
    raise ValueError(f"unknown segmentation model: {name}")


def enable_mc_dropout(model: nn.Module) -> int:
    """Put the model in eval mode but keep dropout layers sampling. Returns their count."""
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
            n += 1
    return n
