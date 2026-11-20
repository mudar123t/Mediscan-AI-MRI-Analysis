"""Fit routing/referral policies on validation data and evaluate them on any test frame.

Used by EXP-006/007 (in-distribution), EXP-008 (synthetic shift) and EXP-009 (external).
A *frame* is a DataFrame with one row per slice:
    y                       true class index
    l_<view>_<c>            raw classifier logits, view in VIEWS, class c in 0..2
    <method>:<score>        segmentation-uncertainty scores (higher = less trustworthy)
Everything fitted (temperatures, tau1, combined-ranking regression, conformal thresholds)
comes from one fold's **val** frame; the test frame is never used for fitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mediscan.calibration import fit_temperature
from mediscan.metrics import cls as clsm
from mediscan.metrics import selective as sel
from mediscan.router import conformal_threshold, fit_tau1, prediction_sets, route
from mediscan.stats import paired_stat_bootstrap
from mediscan.uncertainty import SCORE_NAMES

VIEWS = {"full": ("exp003_r18_full", "gt"), "crop32": ("exp003_r18_crop32", "pred"), "masked": ("exp003_r18_masked", "pred")}
UNC_RUN = "exp005_uncertainty"
ALPHAS = (0.05, 0.10)
COVERAGES = (0.95, 0.90, 0.80)
CLASS_NAMES = ("meningioma", "glioma", "pituitary")


def logits(df: pd.DataFrame, v: str) -> np.ndarray:
    return df[[f"l_{v}_{c}" for c in range(3)]].to_numpy()


def load_fold_frames(k: int, runs: Path, methods: list[str]) -> dict[str, pd.DataFrame]:
    """EXP-003 logits + EXP-005 scores of fold k as val and test frames."""
    scores = None
    for m in methods:
        f = runs / UNC_RUN / f"scores_fold{k}_{m}.csv"
        if f.exists():
            s = pd.read_csv(f)
            s["maxprob_ring"] = -s["maxprob_ring"]
            s = s.rename(columns={c: f"{m}:{c}" for c in SCORE_NAMES})
            scores = s if scores is None else scores.merge(s, on=["row", "part"])
    out = {}
    for part in ("val", "test"):
        df = None
        for v, (run, src) in VIEWS.items():
            z = np.load(runs / run / f"fold{k}" / f"logits_{part}_{src}.npz")
            t = pd.DataFrame({"row": z["rows"], "y": z["labels"], **{f"l_{v}_{c}": z["logits"][:, c] for c in range(3)}})
            df = t if df is None else df.merge(t, on=["row", "y"])
        df = df.merge(scores[scores.part == part].drop(columns="part"), on="row")
        df["fold"] = k
        out[part] = df
    return out


def _combined_features(p: np.ndarray, u: np.ndarray, used: np.ndarray) -> np.ndarray:
    """[-log(1 - MSP), u, used-crop]; the log spreads MSP values that crowd near 1."""
    return np.column_stack([-np.log1p(-np.clip(p.max(1), 0, 1 - 1e-7)), u, used.astype(float)])


def fit_params(va: pd.DataFrame, score: str) -> dict[str, Any]:
    """Fit every policy parameter on one fold's validation frame."""
    fill = float(va[score].max())
    u = va[score].fillna(fill).to_numpy()
    y = va.y.to_numpy()
    temps = {v: fit_temperature(logits(va, v), y) for v in VIEWS}
    P = {v: clsm.softmax(logits(va, v) / temps[v]) for v in VIEWS}
    tau1 = fit_tau1(P["crop32"], P["full"], u, y)
    p_r, used = route(P["crop32"], P["full"], u, tau1)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(_combined_features(p_r, u, used), (p_r.argmax(1) == y).astype(int))
    q = {}
    for alpha in ALPHAS:
        for name, p in (("router", p_r), ("crop32", P["crop32"]), ("full", P["full"])):
            q[f"{name}@{alpha}"] = conformal_threshold(p, y, alpha)
    return {"score": score, "fill": fill, "temps": temps, "tau1": tau1, "lr": lr, "q": q,
            "frac_crop_val": float(used.mean())}


def apply_params(params: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Add calibrated probabilities, routed outputs and conformal sets to a test frame."""
    df = df.copy()
    u = df[params["score"]].fillna(params["fill"]).to_numpy()
    P = {v: clsm.softmax(logits(df, v) / params["temps"][v]) for v in VIEWS}
    for v in VIEWS:
        df[f"pred_{v}"], df[f"conf_{v}"] = P[v].argmax(1), P[v].max(1)
    p_r, used = route(P["crop32"], P["full"], u, params["tau1"])
    df["u"], df["used_crop"] = u, used
    df["pred_router"], df["conf_router"] = p_r.argmax(1), p_r.max(1)
    df["conf_combined"] = params["lr"].predict_proba(_combined_features(p_r, u, used))[:, 1]
    y = df.y.to_numpy()
    crop_ok, full_ok = P["crop32"].argmax(1) == y, P["full"].argmax(1) == y
    p_or = np.where((crop_ok | ~full_ok)[:, None], P["crop32"], P["full"])
    df["pred_oracle"], df["conf_oracle"] = p_or.argmax(1), p_or.max(1)
    for alpha in ALPHAS:
        for name, p in (("router", p_r), ("crop32", P["crop32"]), ("full", P["full"])):
            sets = prediction_sets(p, params["q"][f"{name}@{alpha}"])
            df[f"cp{alpha}_{name}_size"] = sets.sum(1)
            df[f"cp{alpha}_{name}_covered"] = sets[np.arange(len(df)), y]
    return df


def policies(te: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(prediction, referral confidence) per policy; higher confidence = kept first."""
    return {
        "full+MSP": (te.pred_full.to_numpy(), te.conf_full.to_numpy()),
        "masked+MSP": (te.pred_masked.to_numpy(), te.conf_masked.to_numpy()),
        "crop32+MSP": (te.pred_crop32.to_numpy(), te.conf_crop32.to_numpy()),
        "crop32+seg-unc": (te.pred_crop32.to_numpy(), -te.u.to_numpy()),
        "router": (te.pred_router.to_numpy(), te.conf_router.to_numpy()),
        "router+combined": (te.pred_router.to_numpy(), te.conf_combined.to_numpy()),
        "oracle-router": (te.pred_oracle.to_numpy(), te.conf_oracle.to_numpy()),
    }


def bootstrap_policies(te: pd.DataFrame, pids: np.ndarray, n_boot: int = 2000) -> dict:
    """Paired patient-bootstrap CIs of every policy metric and of router-minus-baseline differences."""
    pol = policies(te)
    y = te.y.to_numpy()

    def stat(idx: np.ndarray) -> dict[str, float]:
        out = {}
        for name, (pred, c) in pol.items():
            ok = pred[idx] == y[idx]
            out[f"{name}|bal_acc@100"] = float(clsm.balanced_accuracy_c(clsm.confusion(y[idx], pred[idx])))
            out[f"{name}|aurc"] = sel.aurc(c[idx], ok)
            out[f"{name}|augrc"] = sel.augrc(c[idx], ok)
            out[f"{name}|e_aurc"] = sel.e_aurc(c[idx], ok)
            for cov in COVERAGES:
                out[f"{name}|acc@{int(cov * 100)}"] = 1 - sel.risk_at_coverage(c[idx], ok, cov)
        for base in ("crop32+MSP", "full+MSP"):
            for m in ("bal_acc@100", "aurc", "augrc", "acc@90"):
                for name in ("router", "router+combined"):
                    out[f"{name} - {base}|{m}"] = out[f"{name}|{m}"] - out[f"{base}|{m}"]
        return out

    return paired_stat_bootstrap(pids, stat, n_boot=n_boot)


def conformal_summary(te: pd.DataFrame) -> dict:
    y = te.y.to_numpy()
    cp = {}
    for alpha in ALPHAS:
        for name in ("router", "crop32", "full"):
            size = te[f"cp{alpha}_{name}_size"].to_numpy()
            cov = te[f"cp{alpha}_{name}_covered"].to_numpy().astype(bool)
            single = size == 1
            cp[f"{name}@alpha={alpha}"] = {
                "coverage": float(cov.mean()),
                "coverage_per_class": {c: float(cov[y == i].mean()) for i, c in enumerate(CLASS_NAMES)},
                "mean_set_size": float(size.mean()), "frac_singleton": float(single.mean()),
                "frac_empty": float((size == 0).mean()), "referral_rate": float((~single).mean()),
                "acc_on_singletons": float(cov[single].mean()) if single.any() else float("nan"),
            }
    return cp
