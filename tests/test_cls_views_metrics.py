"""Tests for classifier views, classification metrics, calibration and the confusion bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from mediscan import views
from mediscan.calibration import fit_temperature
from mediscan.metrics import cls as clsm
from mediscan.models.cls import build_cls_model
from mediscan.stats import confusion_bootstrap, per_patient_confusion


def _img_mask(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    img = rng.integers(1, 255, (size, size), dtype=np.uint8)
    mask = np.zeros((size, size), np.uint8)
    mask[20:30, 25:31] = 1  # 10 x 6 box
    return img, mask


def test_full_and_masked_roi() -> None:
    img, mask = _img_mask()
    assert views.full(img, mask) is img
    out = views.masked_roi(img, mask)
    assert (out[mask == 0] == 0).all() and (out[mask == 1] == img[mask == 1]).all()


def test_crop_box_square_contains_mask_and_grows_with_margin() -> None:
    _, mask = _img_mask()
    sides = []
    for m in (0, 8, 16):
        y0, x0, y1, x1 = views.crop_box(mask, m)
        assert y1 - y0 == x1 - x0
        assert y0 <= 20 and y1 >= 30 and x0 <= 25 and x1 >= 31
        assert 0 <= y0 and 0 <= x0 and y1 <= 64 and x1 <= 64
        sides.append(y1 - y0)
    assert sides == [10, 26, 42]


def test_crop_box_clipped_to_image() -> None:
    mask = np.zeros((64, 64), np.uint8)
    mask[0:5, 60:64] = 1
    y0, x0, y1, x1 = views.crop_box(mask, 32)
    assert (y0, x0) >= (0, 0) and y1 <= 64 and x1 <= 64 and y1 - y0 == x1 - x0


def test_context_crop_shape_and_empty_fallback() -> None:
    img, mask = _img_mask()
    assert views.context_crop(img, mask, 8).shape == img.shape
    assert views.context_crop(img, np.zeros_like(mask), 8) is img
    with pytest.raises(ValueError):
        views.build_view("context_crop")


def test_count_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, 300))
    c = clsm.confusion(y, pred)
    assert clsm.balanced_accuracy_c(c) == pytest.approx(balanced_accuracy_score(y, pred))
    assert clsm.macro_f1_c(c) == pytest.approx(f1_score(y, pred, average="macro"))
    assert clsm.mcc_c(c) == pytest.approx(matthews_corrcoef(y, pred))


def test_ece_and_temperature_scaling() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(0, 1, (2000, 3))
    logits[np.arange(2000), y] += 1.0
    hot = logits * 5  # over-confident version of the same predictions
    t = fit_temperature(hot, y)
    assert t > 2.0
    p_hot, p_cal = clsm.softmax(hot), clsm.softmax(hot / t)
    assert clsm.ece(p_cal, y) < clsm.ece(p_hot, y)
    assert clsm.nll(p_cal, y) < clsm.nll(p_hot, y)


def test_confusion_bootstrap_paired() -> None:
    rng = np.random.default_rng(3)
    pids = np.repeat(np.arange(40), 10)
    y = rng.integers(0, 3, 400)
    good = np.where(rng.random(400) < 0.9, y, (y + 1) % 3)
    bad = np.where(rng.random(400) < 0.6, y, (y + 1) % 3)
    _, cg = per_patient_confusion(pids, y, good)
    _, cb = per_patient_confusion(pids, y, bad)
    out = confusion_bootstrap({"bad": cb, "good": cg}, clsm.balanced_accuracy_c, n_boot=500)
    s = out["single"]["good"]
    assert s["ci_low"] <= s["estimate"] <= s["ci_high"]
    assert s["estimate"] == pytest.approx(balanced_accuracy_score(y, good))
    d = out["diff"]["good - bad"]
    assert d["diff"] > 0 and d["ci_low"] > 0 and d["p_two_sided"] < 0.05


@pytest.mark.parametrize("name", ["simplecnn", "resnet18", "efficientnet_b0"])
def test_cls_models_shapes(name: str) -> None:
    model = build_cls_model(name, pretrained=False)
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 1, 256, 256)).shape == (2, 3)
