"""Optimizer and learning-rate schedule from a ``train`` config block (shared by seg and cls)."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch


def make_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name, lr, wd = cfg["optimizer"], float(cfg["lr"]), float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(name)


def lr_lambda(cfg: dict[str, Any], steps_per_epoch: int) -> Callable[[int], float]:
    """Constant (``scheduler: none``) or linear warmup + cosine decay, per optimizer step."""
    total = cfg["epochs"] * steps_per_epoch
    warm = int(cfg.get("warmup_epochs", 0)) * steps_per_epoch
    if cfg.get("scheduler", "none") == "none":
        return lambda step: 1.0

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return f
