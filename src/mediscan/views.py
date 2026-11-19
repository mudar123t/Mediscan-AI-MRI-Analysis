"""Classifier input views built from an image and a (GT or predicted) tumor mask.

All views return a uint8 image of the same size as the input (256² here), so
every classifier sees the same input shape.

* ``full``: the image unchanged (no mask needed).
* ``masked_roi``: semester-1 style. The full frame with every non-tumor pixel set
  to 0 (``legacy/semester1/scripts/07_build_roi_dataset.py::apply_roi``).
* ``context_crop(margin)``: the mask bounding box grown by ``margin`` pixels on
  each side, made square (around its centre), clipped to the image, then resized
  back to the full size. ``margin=0`` is the tight crop.

Empty mask (possible only for predicted masks, 0.13 % of slices in EXP-002c):
``masked_roi`` returns an all-zero image and ``context_crop`` falls back to the
full image. Both are documented in EXP-003.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import cv2
import numpy as np

ViewFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def full(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    return image


def masked_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[mask == 0] = 0
    return out


def crop_box(mask: np.ndarray, margin: int) -> tuple[int, int, int, int] | None:
    """Square box (y0, x0, y1, x1), end-exclusive, around the mask plus margin; None if empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    y0, y1 = ys.min() - margin, ys.max() + 1 + margin
    x0, x1 = xs.min() - margin, xs.max() + 1 + margin
    side = min(max(y1 - y0, x1 - x0), h, w)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    y0 = int(round(cy - side / 2))
    x0 = int(round(cx - side / 2))
    y0 = min(max(y0, 0), h - side)  # shift (not shrink) the square to stay inside
    x0 = min(max(x0, 0), w - side)
    return y0, x0, y0 + side, x0 + side


def context_crop(image: np.ndarray, mask: np.ndarray, margin: int = 16) -> np.ndarray:
    box = crop_box(mask, margin)
    if box is None:
        return image
    y0, x0, y1, x1 = box
    crop = image[y0:y1, x0:x1]
    h, w = image.shape
    interp = cv2.INTER_AREA if crop.shape[0] > h else cv2.INTER_LINEAR
    return cv2.resize(crop, (w, h), interpolation=interp)


def build_view(name: str, margin: int | None = None) -> ViewFn:
    """View function by name: ``full``, ``masked_roi`` or ``context_crop`` (needs ``margin``)."""
    if name == "full":
        return full
    if name == "masked_roi":
        return masked_roi
    if name == "context_crop":
        if margin is None:
            raise ValueError("context_crop needs a margin")
        return partial(context_crop, margin=int(margin))
    raise ValueError(f"unknown view {name!r}")


def uses_mask(name: str) -> bool:
    return name != "full"
