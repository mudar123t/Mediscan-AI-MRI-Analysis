"""Segmentation-quality-aware routing (R1) and split-conformal referral (R7). Plan §3.4.

Per slice we have calibrated class probabilities from two views (context crop built
from the predicted mask, and the mask-free full image) and a segmentation-uncertainty
score ``u`` (higher = mask less trustworthy).

* Route: ``u <= tau1`` -> crop view, else -> full view.
* Refer: the least confident routed cases (risk-coverage), or, with conformal
  prediction, every case whose prediction set is not a single class.
``tau1`` is chosen on validation data only.
"""

from __future__ import annotations

import numpy as np

from mediscan.metrics.selective import aurc


def route(p_crop: np.ndarray, p_full: np.ndarray, u: np.ndarray, tau1: float) -> tuple[np.ndarray, np.ndarray]:
    """Routed probabilities (N, K) and a boolean array 'used crop view'."""
    use_crop = u <= tau1
    return np.where(use_crop[:, None], p_crop, p_full), use_crop


def fit_tau1(p_crop: np.ndarray, p_full: np.ndarray, u: np.ndarray, y: np.ndarray, grid: int = 51) -> float:
    """tau1 minimising the AURC of the routed classifier (MSP ranking) on validation data.

    Candidates are quantiles 0..1 of ``u`` plus +inf (always crop) and -inf (always full);
    ties are resolved towards using the crop view more often (larger tau1).
    """
    cands = np.unique(np.concatenate([np.quantile(u, np.linspace(0, 1, grid)), [-np.inf, np.inf]]))
    best, best_tau = np.inf, np.inf
    for t in cands[::-1]:
        p, _ = route(p_crop, p_full, u, t)
        a = aurc(p.max(1), p.argmax(1) == y)
        if a < best - 1e-12:
            best, best_tau = a, t
    return float(best_tau)


def conformal_threshold(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """Split-conformal (LAC) threshold q: prediction set = {k : 1 - p_k <= q}."""
    scores = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(p: np.ndarray, q: float) -> np.ndarray:
    """Boolean (N, K) membership of each class in the conformal set."""
    return (1.0 - p) <= q
