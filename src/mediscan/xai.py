"""Grad-CAM attribution and region fractions for EXP-010 (R4, explainability as a diagnostic).

* ``gradcam``: Grad-CAM of the predicted class at a target layer (timm ResNet ``layer4``).
* ``backproject``: put a crop-view map back into original-image coordinates.
* ``region_fractions``: share of a unit-mass map inside a region, in a ring around it, and elsewhere.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy import ndimage

_DISK_CACHE: dict[int, np.ndarray] = {}


def _disk(r: int) -> np.ndarray:
    if r not in _DISK_CACHE:
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        _DISK_CACHE[r] = (x * x + y * y) <= r * r
    return _DISK_CACHE[r]


def gradcam(model: torch.nn.Module, layer: torch.nn.Module, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM maps (B, H, W) for the predicted class, upsampled to the input size, and logits (B, K).

    Maps are the ReLU of the gradient-weighted sum of the layer's activations, **not** normalised.
    """
    acts: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        out.retain_grad()
        acts["a"] = out

    h = layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        pred = logits.argmax(1)
        logits.gather(1, pred[:, None]).sum().backward()
        a, g = acts["a"], acts["a"].grad
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    finally:
        h.remove()
    return cam.detach().cpu().numpy(), logits.detach().cpu().numpy()


def backproject(cam_view: np.ndarray, box: tuple[int, int, int, int] | None, shape: tuple[int, int]) -> np.ndarray:
    """Map a crop-view map back to the original image; ``box`` = (y0, x0, y1, x1) or None (view = full image)."""
    if box is None:
        return cam_view.astype(np.float32)
    y0, x0, y1, x1 = box
    out = np.zeros(shape, np.float32)
    out[y0:y1, x0:x1] = cv2.resize(cam_view.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return out


def region_fractions(cam: np.ndarray, region: np.ndarray, ring_px: int = 16) -> tuple[float, float, float]:
    """(in, ring, out) shares of the map's mass; NaN if the map has no mass or the region is empty."""
    total = float(cam.sum())
    reg = region.astype(bool)
    if total <= 0 or not reg.any():
        return float("nan"), float("nan"), float("nan")
    ring = ndimage.binary_dilation(reg, structure=_disk(ring_px)) & ~reg
    f_in = float(cam[reg].sum()) / total
    f_ring = float(cam[ring].sum()) / total
    return f_in, f_ring, 1.0 - f_in - f_ring
