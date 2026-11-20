"""Tests for selective metrics, routing and conformal referral (EXP-006/007)."""

from __future__ import annotations

import numpy as np
import pytest

from mediscan.metrics import selective as sel
from mediscan.router import conformal_threshold, fit_tau1, prediction_sets, route


def test_aurc_perfect_ranking_and_e_aurc() -> None:
    correct = np.array([1, 1, 1, 0])
    good = np.array([0.9, 0.8, 0.7, 0.1])
    bad = np.array([0.1, 0.8, 0.7, 0.9])
    assert sel.e_aurc(good, correct) == pytest.approx(0.0)
    assert sel.aurc(bad, correct) > sel.aurc(good, correct)
    assert sel.risk_at_coverage(good, correct, 0.75) == 0.0
    cov, risk = sel.risk_coverage(good, correct)
    assert cov[-1] == 1.0 and risk[-1] == pytest.approx(0.25)
    assert sel.augrc(good, correct) == pytest.approx(np.mean([0, 0, 0, 0.25]))


def test_route_and_fit_tau1_prefers_crop_when_mask_good() -> None:
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 3, n)
    u = rng.random(n)
    onehot = np.eye(3)[y]
    wrong = np.eye(3)[(y + 1) % 3]
    # crop view is right when u < 0.5 and wrong otherwise; full view right 80 % everywhere
    p_crop = np.where((u < 0.5)[:, None], 0.9 * onehot + 0.05, 0.9 * wrong + 0.05)
    full_ok = rng.random(n) < 0.8
    p_full = np.where(full_ok[:, None], 0.6 * onehot + 0.2, 0.6 * wrong + 0.2)
    tau = fit_tau1(p_crop, p_full, u, y)
    assert 0.4 <= tau <= 0.6
    p, used = route(p_crop, p_full, u, tau)
    assert used.mean() == pytest.approx((u <= tau).mean())
    assert (p.argmax(1) == y).mean() > max((p_crop.argmax(1) == y).mean(), (p_full.argmax(1) == y).mean())


def test_conformal_coverage_close_to_target() -> None:
    rng = np.random.default_rng(1)
    def draw(n):
        y = rng.integers(0, 3, n)
        logits = rng.normal(0, 1, (n, 3)); logits[np.arange(n), y] += 1.5
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return p, y
    pc, yc = draw(2000)
    pt, yt = draw(4000)
    for alpha in (0.05, 0.1):
        q = conformal_threshold(pc, yc, alpha)
        sets = prediction_sets(pt, q)
        cov = sets[np.arange(len(yt)), yt].mean()
        assert abs(cov - (1 - alpha)) < 0.02
