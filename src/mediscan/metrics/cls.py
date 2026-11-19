"""Classification metrics (plan §2: balanced accuracy, macro-F1, MCC, AUROC, ECE, Brier).

Count-based metrics are computed from a confusion matrix ``C[true, pred]`` so the
patient bootstrap can resample per-patient confusion matrices (``stats``).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def confusion(y: np.ndarray, pred: np.ndarray, k: int = 3) -> np.ndarray:
    c = np.zeros((k, k), dtype=np.int64)
    np.add.at(c, (y, pred), 1)
    return c


def _per_class(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recall and precision per class for a (..., k, k) confusion array (0 where undefined)."""
    tp = np.diagonal(c, axis1=-2, axis2=-1).astype(float)
    support = c.sum(-1).astype(float)
    predicted = c.sum(-2).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    return recall, precision


def accuracy_c(c: np.ndarray) -> np.ndarray:
    return np.trace(c, axis1=-2, axis2=-1) / c.sum((-2, -1))


def balanced_accuracy_c(c: np.ndarray) -> np.ndarray:
    return _per_class(c)[0].mean(-1)


def macro_f1_c(c: np.ndarray) -> np.ndarray:
    r, p = _per_class(c)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return f1.mean(-1)


def mcc_c(c: np.ndarray) -> np.ndarray:
    """Multiclass Matthews correlation (Gorodkin's R_K), vectorised over leading axes."""
    c = c.astype(float)
    t = c.sum(-1)  # true counts
    p = c.sum(-2)  # predicted counts
    n = c.sum((-2, -1))
    correct = np.trace(c, axis1=-2, axis2=-1)
    cov_yp = correct * n - (t * p).sum(-1)
    cov_yy = n**2 - (t * t).sum(-1)
    cov_pp = n**2 - (p * p).sum(-1)
    denom = np.sqrt(cov_yy * cov_pp)
    return np.divide(cov_yp, denom, out=np.zeros_like(cov_yp), where=denom > 0)


COUNT_METRICS = {
    "accuracy": accuracy_c,
    "balanced_accuracy": balanced_accuracy_c,
    "macro_f1": macro_f1_c,
    "mcc": mcc_c,
}


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 confidence, equal-width bins."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(total)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())


def all_metrics(logits: np.ndarray, y: np.ndarray, class_names: list[str]) -> dict:
    """Point estimates of every classification metric for one set of logits."""
    probs = softmax(logits)
    pred = probs.argmax(1)
    c = confusion(y, pred, len(class_names))
    recall, _ = _per_class(c)
    out = {name: float(fn(c)) for name, fn in COUNT_METRICS.items()}
    out["auroc_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    out["ece"] = ece(probs, y)
    out["brier"] = brier(probs, y)
    out["nll"] = nll(probs, y)
    out["recall"] = {n: float(r) for n, r in zip(class_names, recall)}
    out["confusion"] = c.tolist()
    return out
