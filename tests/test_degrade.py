"""Tests for controlled mask degradation (EXP-004)."""

from __future__ import annotations

import numpy as np
import pytest

from mediscan.degrade import KINDS, degrade
from mediscan.metrics.seg import dice


def _disk(r: int = 15, size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (((yy - size // 2) ** 2 + (xx - size // 2) ** 2) <= r * r).astype(np.uint8)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("target", [0.9, 0.6, 0.3])
def test_degrade_hits_target(kind: str, target: float) -> None:
    gt = _disk()
    m, d = degrade(gt, kind, target, np.random.default_rng(0))
    assert d == pytest.approx(dice(m, gt))
    assert abs(d - target) < 0.06
    assert m.dtype == np.uint8 and m.shape == gt.shape


def test_target_one_returns_gt_and_is_deterministic() -> None:
    gt = _disk()
    m, d = degrade(gt, "shift", 1.0, np.random.default_rng(0))
    assert d == 1.0 and (m == gt).all()
    a, _ = degrade(gt, "blob", 0.5, np.random.default_rng(7))
    b, _ = degrade(gt, "blob", 0.5, np.random.default_rng(7))
    assert (a == b).all()


def test_kind_semantics() -> None:
    gt = _disk()
    er, _ = degrade(gt, "erode", 0.6, np.random.default_rng(0))
    di, _ = degrade(gt, "dilate", 0.6, np.random.default_rng(0))
    bl, _ = degrade(gt, "blob", 0.6, np.random.default_rng(0))
    assert er.sum() < gt.sum() and not (er & ~gt.astype(bool)).any()
    assert di.sum() > gt.sum() and (di >= gt).all()
    assert (bl >= gt).all() and bl.sum() > gt.sum()
