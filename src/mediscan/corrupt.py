"""Synthetic acquisition shifts for EXP-008 (plan: noise, blur, contrast/gamma, bias field; 3 severities).

Images are uint8 (H, W); corruption is applied in [0, 1] and converted back to uint8.
Severity levels are fixed here, before any result is seen [I: chosen to range from
mild to strong; not taken from a benchmark]. Randomness comes only from the given generator.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

KINDS = ("noise", "blur", "gamma", "bias")
LEVELS = {
    "noise": (0.03, 0.06, 0.10),   # Gaussian noise std (fraction of the [0, 1] range)
    "blur": (1.0, 2.0, 3.0),       # Gaussian blur sigma (px at 256²)
    "gamma": (1.5, 2.0, 3.0),      # x ** gamma (darker, lower contrast in dark tissue)
    "bias": (0.2, 0.4, 0.6),       # multiplicative smooth field exp(s * g), g in [-1, 1]
}


def corrupt(image: np.ndarray, kind: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Return the corrupted uint8 image; ``severity`` is 1, 2 or 3."""
    x = image.astype(np.float32) / 255.0
    level = LEVELS[kind][severity - 1]
    if kind == "noise":
        x = x + rng.normal(0.0, level, x.shape).astype(np.float32)
    elif kind == "blur":
        x = ndimage.gaussian_filter(x, sigma=level)
    elif kind == "gamma":
        x = np.power(np.clip(x, 0, 1), level)
    elif kind == "bias":
        g = ndimage.gaussian_filter(rng.normal(0, 1, x.shape).astype(np.float32), sigma=40)
        g = g / (np.abs(g).max() + 1e-8)
        x = x * np.exp(level * g)
    else:
        raise ValueError(f"unknown corruption {kind!r}")
    return np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)
