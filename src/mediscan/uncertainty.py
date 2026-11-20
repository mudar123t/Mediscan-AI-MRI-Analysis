"""Segmentation uncertainty: sampling (MC dropout, TTA, seed ensemble) and image-level scores (EXP-005).

Every method yields T probability maps per image. Scores are computed against the
**reference mask** M, the deterministic prediction the classifier actually uses
(threshold 0.5, largest component), so that they describe how trustworthy M is.

Image-level scores (higher = less trustworthy, except ``maxprob_ring``):
* ``ent_ring``: mean binary entropy (bits) of the mean probability over M dilated by
  ``ring`` px (the tumor and its border).
* ``ent_total``: summed entropy over the whole image / 100 (also catches spurious regions).
* ``disagree``: 1 - mean Dice between each sample's mask (p > 0.5) and M.
* ``area_cv``: coefficient of variation of the sample mask areas.
* ``maxprob_ring``: mean of max(p, 1 - p) over the ring region (confidence).
If M is empty, the ring region falls back to pixels with mean probability > 0.1;
if that is empty too, the region-based scores are NaN (filled with the worst value
by the analysis).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from mediscan.metrics.seg import dice

_CROSS = ndimage.generate_binary_structure(2, 1)
SCORE_NAMES = ("ent_ring", "ent_total", "disagree", "area_cv", "maxprob_ring")


def binary_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def image_scores(samples: np.ndarray, ref_mask: np.ndarray, ring: int = 5) -> dict[str, float]:
    """Scores for one image from ``samples`` (T, H, W) probabilities and the reference mask."""
    m = samples.mean(0)
    ent = binary_entropy(m)
    ref = ref_mask.astype(bool)
    region = ndimage.binary_dilation(ref, _CROSS, iterations=ring) if ref.any() else m > 0.1
    out = {"ent_total": float(ent.sum() / 100.0)}
    if region.any():
        out["ent_ring"] = float(ent[region].mean())
        out["maxprob_ring"] = float(np.maximum(m, 1 - m)[region].mean())
    else:
        out["ent_ring"] = out["maxprob_ring"] = float("nan")
    bins = samples > 0.5
    out["disagree"] = float(1.0 - np.mean([dice(b, ref) for b in bins]))
    areas = bins.reshape(len(bins), -1).sum(1).astype(float)
    out["area_cv"] = float(areas.std() / areas.mean()) if areas.mean() > 0 else float("nan")
    return out


@torch.no_grad()
def mc_dropout_samples(model: torch.nn.Module, x: torch.Tensor, t: int, amp: bool) -> torch.Tensor:
    """(T, B, H, W) sigmoid probabilities with dropout active (call ``enable_mc_dropout`` first)."""
    outs = []
    for _ in range(t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            outs.append(torch.sigmoid(model(x).float())[:, 0])
    return torch.stack(outs)


@torch.no_grad()
def tta_samples(model: torch.nn.Module, x: torch.Tensor, amp: bool) -> torch.Tensor:
    """(8, B, H, W) probabilities under the 8 dihedral transforms (4 rotations x optional flip), mapped back."""
    outs = []
    for flip in (False, True):
        xf = torch.flip(x, dims=[-1]) if flip else x
        for k in range(4):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                p = torch.sigmoid(model(torch.rot90(xf, k, dims=[-2, -1])).float())[:, 0]
            p = torch.rot90(p, -k, dims=[-2, -1])
            outs.append(torch.flip(p, dims=[-1]) if flip else p)
    return torch.stack(outs)
