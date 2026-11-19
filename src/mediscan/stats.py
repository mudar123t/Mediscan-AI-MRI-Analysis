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
