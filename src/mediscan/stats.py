"""Patient-level bootstrap confidence intervals and paired comparisons.

Slices from one patient are correlated, so we resample *patients* (with all
their slices), not slices. 2,000 resamples, percentile intervals.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def patient_bootstrap(
    df: pd.DataFrame,
    stat: Callable[[pd.DataFrame], float],
    pid_col: str = "pid",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Point estimate and (1 - alpha) percentile CI of ``stat`` over patient resamples."""
    groups = {pid: g for pid, g in df.groupby(pid_col, sort=True)}
    pids = np.array(list(groups))
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(pids, size=len(pids), replace=True)
        values[b] = stat(pd.concat([groups[p] for p in sample], ignore_index=True))
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(stat(df)), "ci_low": float(lo), "ci_high": float(hi), "n_patients": len(pids)}


def paired_patient_bootstrap(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    pid_col: str = "pid",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """CI of mean(col_a) - mean(col_b) on the same slices, resampling patients.

    ``p_two_sided`` is the bootstrap estimate of P(difference has the opposite
    sign to the observed one), doubled and capped at 1.
    """
    per_pid = df.groupby(pid_col).agg(a_sum=(col_a, "sum"), b_sum=(col_b, "sum"), n=(col_a, "size"))
    a, b, n = per_pid["a_sum"].to_numpy(), per_pid["b_sum"].to_numpy(), per_pid["n"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_pid), size=(n_boot, len(per_pid)))
    diffs = (a[idx].sum(1) - b[idx].sum(1)) / n[idx].sum(1)
    observed = float((a.sum() - b.sum()) / n.sum())
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if observed >= 0:
        p = 2 * float(np.mean(diffs <= 0))
    else:
        p = 2 * float(np.mean(diffs >= 0))
    return {"diff": observed, "ci_low": float(lo), "ci_high": float(hi), "p_two_sided": min(p, 1.0)}


def mean_ci(
    df: pd.DataFrame,
    col: str,
    pid_col: str = "pid",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Patient-bootstrap CI of the slice-level mean of one column (vectorised).

    Equivalent to ``patient_bootstrap(df, lambda d: d[col].mean())``.
    """
    per_pid = df.groupby(pid_col)[col].agg(["sum", "size"])
    s, n = per_pid["sum"].to_numpy(dtype=float), per_pid["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_pid), size=(n_boot, len(per_pid)))
    values = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(df[col].mean()), "ci_low": float(lo), "ci_high": float(hi), "n_patients": len(per_pid)}


def per_patient_confusion(pids: np.ndarray, y: np.ndarray, pred: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique patient ids and their (P, k, k) confusion matrices C[true, pred]."""
    uniq, inv = np.unique(np.asarray(pids), return_inverse=True)
    conf = np.zeros((len(uniq), k, k), dtype=np.int64)
    np.add.at(conf, (inv, y, pred), 1)
    return uniq, conf


def confusion_bootstrap(
    confs: dict[str, np.ndarray],
    metric: Callable[[np.ndarray], np.ndarray],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Patient-bootstrap CIs of a confusion-matrix metric for several systems at once.

    ``confs`` maps a system name to its (P, k, k) per-patient confusion array; all
    arrays must be over the same patients in the same order. The same patient
    resamples are used for every system, so ``diff`` CIs are paired.
    Returns ``{"single": {name: ci}, "diff": {"b - a": ci}}`` (``diff`` for every pair).
    """
    names = list(confs)
    n_pat = len(next(iter(confs.values())))
    if any(len(c) != n_pat for c in confs.values()):
        raise ValueError("systems cover different patients")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pat, size=(n_boot, n_pat))
    boot = {n: metric(confs[n][idx].sum(1)) for n in names}
    point = {n: float(metric(confs[n].sum(0))) for n in names}
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    out: dict = {"single": {}, "diff": {}}
    for n in names:
        lo, hi = np.percentile(boot[n], q)
        out["single"][n] = {"estimate": point[n], "ci_low": float(lo), "ci_high": float(hi), "n_patients": n_pat}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = boot[b] - boot[a]
            obs = point[b] - point[a]
            lo, hi = np.percentile(d, q)
            p = 2 * float(np.mean(d <= 0)) if obs >= 0 else 2 * float(np.mean(d >= 0))
            out["diff"][f"{b} - {a}"] = {"diff": obs, "ci_low": float(lo), "ci_high": float(hi), "p_two_sided": min(p, 1.0)}
    return out


def patient_resamples(pids: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Yield ``n_boot`` slice-index arrays, each a resample of patients (with all their slices)."""
    pids = np.asarray(pids)
    uniq, inv = np.unique(pids, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        yield np.concatenate([groups[i] for i in pick])


def paired_stat_bootstrap(
    pids: np.ndarray,
    stat: Callable[[np.ndarray], dict[str, float]],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Patient-bootstrap CIs for several statistics computed on the same resamples.

    ``stat(idx)`` returns ``{name: value}`` for the slices ``idx``; resamples where
    a statistic is undefined (NaN, e.g. AUROC with one class) are dropped for it.
    """
    point = stat(np.arange(len(pids)))
    boot: dict[str, list[float]] = {k: [] for k in point}
    for idx in patient_resamples(pids, n_boot, seed):
        for k, v in stat(idx).items():
            boot[k].append(v)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    out = {}
    for k, v in point.items():
        b = np.asarray(boot[k], dtype=float)
        b = b[~np.isnan(b)]
        lo, hi = np.percentile(b, q) if b.size else (np.nan, np.nan)
        out[k] = {"estimate": float(v), "ci_low": float(lo), "ci_high": float(hi)}
    return out
