"""Tests for segmentation-uncertainty sampling and scores (EXP-005)."""

from __future__ import annotations

import numpy as np
import torch

from mediscan.models.seg import build_seg_model, enable_mc_dropout
from mediscan.uncertainty import SCORE_NAMES, image_scores, mc_dropout_samples, tta_samples


class _PixelModel(torch.nn.Module):
    """Position-dependent toy 'segmenter': logit = 4 * (x - 0.5); equivariant to rotations/flips."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 4 * (x - 0.5)


def _disk() -> np.ndarray:
    yy, xx = np.mgrid[0:64, 0:64]
    return (((yy - 30) ** 2 + (xx - 20) ** 2) <= 100).astype(np.uint8)


def test_tta_maps_back_to_input_frame() -> None:
    x = torch.from_numpy(_disk().astype(np.float32))[None, None]
    s = tta_samples(_PixelModel(), x, amp=False)
    assert s.shape == (8, 1, 64, 64)
    ref = torch.sigmoid(4 * (x[:, 0] - 0.5))
    for t in range(8):
        assert torch.allclose(s[t], ref)


def test_scores_rank_certain_below_uncertain() -> None:
    gt = _disk()
    sure = np.stack([gt * 0.98 + 0.01] * 6)
    rng = np.random.default_rng(0)
    shaky = np.stack([np.roll(gt, int(rng.integers(-6, 7)), axis=1) * 0.7 + 0.1 for _ in range(6)])
    a, b = image_scores(sure, gt), image_scores(shaky, gt)
    assert set(SCORE_NAMES) <= set(a)
    assert b["ent_ring"] > a["ent_ring"] and b["disagree"] > a["disagree"] and b["maxprob_ring"] < a["maxprob_ring"]
    assert a["disagree"] == 0.0 and a["area_cv"] == 0.0


def test_empty_reference_mask_gives_nan_region_scores() -> None:
    s = np.full((3, 64, 64), 0.01, dtype=np.float32)
    out = image_scores(s, np.zeros((64, 64), np.uint8))
    assert np.isnan(out["ent_ring"]) and np.isnan(out["area_cv"])


def test_mc_dropout_sampling_varies() -> None:
    model = build_seg_model("unet_r34", pretrained=False, dropout=0.5)
    enable_mc_dropout(model)
    s = mc_dropout_samples(model, torch.rand(2, 1, 64, 64), t=4, amp=False)
    assert s.shape == (4, 2, 64, 64) and not torch.allclose(s[0], s[1])
