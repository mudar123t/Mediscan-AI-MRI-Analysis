"""Controlled degradation of ground-truth masks to a target Dice (EXP-004, R3 study).

Four failure types that a segmenter makes:

* ``erode``: under-segmentation (k iterations of 4-connected erosion);
* ``dilate``: over-segmentation (k iterations of 4-connected dilation);
* ``shift``: misplacement (translation by d px in a random direction, zero fill);
* ``blob``: a false-positive disk inside the head, away from the tumor (the GT
  mask is kept, so a bounding-box crop grows to include the blob).

For each type the free parameter is searched so that Dice(degraded, GT) is as
close as possible to the target. The achieved Dice is returned and reported;
targets a type cannot reach (e.g. erosion of a tiny mask) give the closest value.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mediscan.metrics.seg import dice

KINDS = ("erode", "dilate", "shift", "blob")
_CROSS = ndimage.generate_binary_structure(2, 1)


def _closest(candidates: list[tuple[float, np.ndarray]], target: float) -> tuple[np.ndarray, float]:
    d, m = min(candidates, key=lambda c: abs(c[0] - target))
    return m, d


def _morph(gt: np.ndarray, target: float, op, max_iter: int = 64) -> tuple[np.ndarray, float]:
    cands = [(1.0, gt)]
    cur = gt.astype(bool)
    for _ in range(max_iter):
        cur = op(cur, structure=_CROSS)
        d = dice(cur, gt)
        cands.append((d, cur.astype(np.uint8)))
        if d <= target or not cur.any():
            break
    return _closest(cands, target)


def _shift(gt: np.ndarray, target: float, rng: np.random.Generator, max_shift: int = 128) -> tuple[np.ndarray, float]:
    theta = rng.uniform(0, 2 * np.pi)
    cands = [(1.0, gt)]
    for s in range(1, max_shift + 1):
        dy, dx = int(round(s * np.sin(theta))), int(round(s * np.cos(theta)))
        out = np.zeros_like(gt)
        h, w = gt.shape
        ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        out[yd, xd] = gt[ys, xs]
        d = dice(out, gt)
        cands.append((d, out))
        if d <= target:
            break
    return _closest(cands, target)


def _blob(gt: np.ndarray, target: float, image: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Add one disk of area B so that Dice = 2|G| / (2|G| + B) ≈ target."""
    g = int(gt.astype(bool).sum())
    if target >= 1.0 or g == 0:
        return gt, 1.0
    area = 2 * g * (1.0 / target - 1.0)
    r = max(1.0, float(np.sqrt(area / np.pi)))
    far = ndimage.distance_transform_edt(~gt.astype(bool)) > r + 3
    h, w = gt.shape
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (yy >= r) & (yy < h - r) & (xx >= r) & (xx < w - r)
    allowed = far & inside
    if image is not None:
        head = image > 0.1 * 255  # rough head region, so blobs land on tissue, not air
        if (allowed & head).any():
            allowed &= head
    if not allowed.any():
        allowed = far if far.any() else np.ones_like(far)
    cy, cx = np.argwhere(allowed)[rng.integers(0, int(allowed.sum()))]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    out = (gt.astype(bool) | disk).astype(np.uint8)
    return out, dice(out, gt)


def degrade(
    gt: np.ndarray, kind: str, target: float, rng: np.random.Generator, image: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Degrade ``gt`` with failure type ``kind`` towards Dice ``target``; returns (mask uint8, achieved Dice)."""
    gt = gt.astype(np.uint8)
    if target >= 1.0:
        return gt, 1.0
    if kind == "erode":
        return _morph(gt, target, ndimage.binary_erosion)
    if kind == "dilate":
        return _morph(gt, target, ndimage.binary_dilation)
    if kind == "shift":
        return _shift(gt, target, rng)
    if kind == "blob":
        return _blob(gt, target, image, rng)
    raise ValueError(f"unknown degradation {kind!r}")
