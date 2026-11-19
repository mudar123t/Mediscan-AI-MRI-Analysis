"""Tests for segmentation metrics, bootstrap statistics and model factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mediscan.metrics import seg as m
from mediscan.models.seg import build_seg_model, enable_mc_dropout
from mediscan.stats import mean_ci, paired_patient_bootstrap


def _square(y0: int, x0: int, s: int, size: int = 64) -> np.ndarray:
    a = np.zeros((size, size), np.uint8)
    a[y0 : y0 + s, x0 : x0 + s] = 1
    return a


def test_overlap_metrics() -> None:
    a, b = _square(10, 10, 10), _square(10, 15, 10)  # overlap 10x5
    assert m.dice(a, a) == 1.0 and m.iou(a, a) == 1.0
    assert m.dice(a, b) == pytest.approx(2 * 50 / 200)
    assert m.iou(a, b) == pytest.approx(50 / 150)


def test_boundary_metrics_on_shift() -> None:
    a, b = _square(10, 10, 10), _square(10, 13, 10)  # shifted by 3 px
    assert m.hd95(a, a) == 0.0 and m.nsd(a, a) == 1.0
    assert m.hd95(a, b) == pytest.approx(3.0)
    assert 0.0 < m.nsd(a, b, tau=2) < 1.0
    assert m.nsd(a, b, tau=3) == 1.0


def test_empty_prediction_rule() -> None:
    gt, empty = _square(10, 10, 10), np.zeros((64, 64), np.uint8)
    assert m.dice(empty, gt) == 0.0 and m.nsd(empty, gt) == 0.0
    assert m.hd95(empty, gt) == pytest.approx(np.hypot(64, 64))


def test_largest_component() -> None:
    a = _square(5, 5, 10) | _square(40, 40, 3)
    out = m.largest_component(a)
    assert out.sum() == 100


def test_bootstrap_ci_contains_estimate() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pid": np.repeat(np.arange(30), 4), "x": rng.random(120), "y": rng.random(120) + 0.2})
    ci = mean_ci(df, "x", n_boot=300)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"] and ci["n_patients"] == 30
    d = paired_patient_bootstrap(df, "y", "x", n_boot=300)
    assert d["diff"] > 0 and d["ci_low"] > 0 and d["p_two_sided"] < 0.05


@pytest.mark.parametrize("name", ["unet_s1", "unet_r34"])
def test_seg_models_shapes(name: str) -> None:
    kwargs = {"pretrained": False} if name == "unet_r34" else {}
    model = build_seg_model(name, **kwargs)
    assert model(torch.zeros(2, 1, 64, 64)).shape == (2, 1, 64, 64)


def test_mc_dropout_is_stochastic() -> None:
    model = build_seg_model("unet_r34", pretrained=False, dropout=0.5)
    assert enable_mc_dropout(model) == 5
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        assert not torch.allclose(model(x), model(x))
