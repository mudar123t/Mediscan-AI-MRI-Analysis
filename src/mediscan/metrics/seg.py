"""Per-image segmentation metrics (Metrics Reloaded selection: overlap + boundary).

All distances are in **pixels** of the evaluated grid (the Figshare files have
no pixel spacing; EXP-001). Every image in this dataset has a non-empty ground
truth, so the only special case is an empty prediction:

* empty prediction, non-empty GT -> Dice 0, IoU 0, NSD 0, HD95 = image diagonal
  (the largest possible distance, a fixed worst-case penalty);
* both empty -> Dice 1, IoU 1, NSD 1, HD95 0 (kept for completeness).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(p, g).sum() / denom)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, g).sum() / union)


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels: mask minus its 4-connected erosion."""
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1), border_value=0)
    return np.logical_and(mask, ~eroded)


def _surface_distances(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distances from surface of ``a`` to surface of ``b`` and vice versa (pixels)."""
    sa, sb = _surface(a), _surface(b)
    dt_b = ndimage.distance_transform_edt(~sb)
    dt_a = ndimage.distance_transform_edt(~sa)
    return dt_b[sa], dt_a[sb]


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance in pixels."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float(np.hypot(*p.shape))
    d_pg, d_gp = _surface_distances(p, g)
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def nsd(pred: np.ndarray, gt: np.ndarray, tau: float = 2.0) -> float:
    """Normalised surface distance (surface Dice) at tolerance ``tau`` pixels."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    d_pg, d_gp = _surface_distances(p, g)
    return float(((d_pg <= tau).sum() + (d_gp <= tau).sum()) / (d_pg.size + d_gp.size))


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component (post-processing ablation)."""
    labels, n = ndimage.label(mask.astype(bool), structure=np.ones((3, 3)))
    if n <= 1:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    return (labels == (int(np.argmax(sizes)) + 1)).astype(np.uint8)


def all_metrics(pred: np.ndarray, gt: np.ndarray, tau: float = 2.0) -> dict[str, float]:
    """Dice, IoU, HD95 and NSD for one image, plus predicted/GT areas."""
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "hd95": hd95(pred, gt),
        "nsd": nsd(pred, gt, tau),
        "pred_area": int(pred.astype(bool).sum()),
        "gt_area": int(gt.astype(bool).sum()),
    }
