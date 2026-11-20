"""Selective classification (referral) metrics.

A system outputs a prediction and a confidence per case; the least confident cases are
referred to a human. Sorting by confidence (descending) gives the risk-coverage curve.

* ``aurc``: area under the risk-coverage curve (Geifman & El-Yaniv 2017), the mean of
  the selective risk over all coverage levels 1/n .. n/n;
* ``e_aurc``: AURC minus the AURC of an oracle ranking with the same errors;
* ``augrc``: area under the *generalised* risk-coverage curve, i.e. the mean over
  coverage levels of P(accepted and wrong) (Traub et al. 2024);
* ``risk_at_coverage``: selective error when the most confident fraction c is kept.
Ties in confidence are broken by the stable sort order (input order).
"""

from __future__ import annotations

import numpy as np


def _sorted_errors(confidence: np.ndarray, correct: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="stable")
    return 1.0 - np.asarray(correct, dtype=float)[order]


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage levels k/n and the selective risk of the top-k most confident cases."""
    err = _sorted_errors(confidence, correct)
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    return float(risk_coverage(confidence, correct)[1].mean())


def e_aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=float)
    return aurc(confidence, correct) - aurc(correct, correct)  # oracle: all correct cases first


def augrc(confidence: np.ndarray, correct: np.ndarray) -> float:
    err = _sorted_errors(confidence, correct)
    return float((np.cumsum(err) / len(err)).mean())


def risk_at_coverage(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    err = _sorted_errors(confidence, correct)
    k = max(1, int(round(coverage * len(err))))
    return float(err[:k].mean())
