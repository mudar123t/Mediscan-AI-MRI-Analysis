"""EXP-006 (R1 routing) and EXP-007 (R7 conformal referral) on the pooled test folds.

Inputs (all existing; no training): EXP-003 view logits (crop 32 / masked on predicted
masks, full image), EXP-005 uncertainty scores, per fold. Everything that is fitted
(temperatures, the uncertainty score used, tau1, the combined-ranking regression,
conformal thresholds) is fitted per fold on that fold's **val** slices only.

Policies (prediction, confidence used for referral):
* ``full+MSP``, ``crop32+MSP``, ``masked+MSP``: single view, ranked by its own MSP
  (``crop32+MSP`` is the classifier-only referral baseline, Sharma-style MSP gate);
* ``crop32+seg-unc``: crop-32 prediction ranked by segmentation uncertainty only;
* ``router``: view chosen by u <= tau1, ranked by the routed MSP (R1);
* ``router+combined``: same routing, ranked by a val-fitted logistic regression on
  [routed MSP, u, used-crop flag];
* ``oracle-router``: picks the view that is correct (crop if both are), ranked by routed MSP
  (upper bound of view selection).

Usage:
    .venv/Scripts/python scripts/eval_exp006.py --out project-log/experiments/EXP-006-router [--score mc:ent_ring]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mediscan.calibration import fit_temperature  # noqa: E402
from mediscan.metrics import cls as clsm  # noqa: E402
from mediscan.metrics import selective as sel  # noqa: E402
from mediscan.router import conformal_threshold, fit_tau1, prediction_sets, route  # noqa: E402
from mediscan.stats import paired_stat_bootstrap  # noqa: E402
from mediscan.uncertainty import SCORE_NAMES  # noqa: E402

VIEWS = {"full": ("exp003_r18_full", "gt"), "crop32": ("exp003_r18_crop32", "pred"), "masked": ("exp003_r18_masked", "pred")}
UNC_RUN = "exp005_uncertainty"
FOLDS = range(5)
COVERAGES = (0.95, 0.90, 0.80)
POLICY_COLORS = {"full+MSP": "#6b6a65", "crop32+MSP": "#2a78d6", "router": "#eb6834", "router+combined": "#1baf7a",
                 "oracle-router": "#9a9893"}


def load_fold(k: int, runs: Path, methods: list[str]) -> dict[str, pd.DataFrame]:
    """Per part (val/test): rows, labels, calibrated probabilities per view, uncertainty scores."""
    out = {}
    scores = None
    for m in methods:
        f = runs / UNC_RUN / f"scores_fold{k}_{m}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        s["maxprob_ring"] = -s["maxprob_ring"]
        s = s.rename(columns={c: f"{m}:{c}" for c in SCORE_NAMES})
        scores = s if scores is None else scores.merge(s, on=["row", "part"])
    temps = {}
    for part in ("val", "test"):
        df = None
        for v, (run, src) in VIEWS.items():
            z = np.load(runs / run / f"fold{k}" / f"logits_{part}_{src}.npz")
            if part == "val":
                temps[v] = fit_temperature(z["logits"], z["labels"])
            p = clsm.softmax(z["logits"] / temps[v])
            t = pd.DataFrame({"row": z["rows"], "y": z["labels"], **{f"p_{v}_{c}": p[:, c] for c in range(3)}})
            df = t if df is None else df.merge(t, on=["row", "y"])
        df = df.merge(scores[scores.part == part].drop(columns="part"), on="row")
        df["fold"] = k
        out[part] = df
    out["temps"] = temps
    return out


def P(df: pd.DataFrame, v: str) -> np.ndarray:
    return df[[f"p_{v}_{c}" for c in range(3)]].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="*", default=["det", "mc", "tta", "ens"])
    ap.add_argument("--score", default=None, help="force the uncertainty score (default: best val AUROC)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    runs = Path("runs")
    meta = pd.read_csv("data/processed/figshare_256/metadata.csv", dtype={"pid": str})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = {k: load_fold(k, runs, args.methods) for k in FOLDS}
    score_cols = [c for c in folds[0]["val"] if ":" in c]

    # ---- 1. choose the uncertainty score on VAL only: AUROC for crop-32 error ---------------
    val = pd.concat([folds[k]["val"] for k in FOLDS], ignore_index=True)
    for c in score_cols:
        val[c] = val[c].fillna(val[c].max())
    err_val = (P(val, "crop32").argmax(1) != val.y).astype(int)
    val_auc = {c: float(roc_auc_score(err_val, val[c])) for c in score_cols}
    score = args.score or max(val_auc, key=val_auc.get)
    print(f"uncertainty score: {score} (val AUROC for crop-32 error {val_auc[score]:.3f})")

    # ---- 2. per fold: fit tau1, combined ranking, conformal thresholds on val; apply to test --
    tests, fitted = [], {}
    for k in FOLDS:
        va, te = folds[k]["val"].copy(), folds[k]["test"].copy()
        for d in (va, te):
            d[score] = d[score].fillna(va[score].max())
        tau1 = fit_tau1(P(va, "crop32"), P(va, "full"), va[score].to_numpy(), va.y.to_numpy())
        p_va, used_va = route(P(va, "crop32"), P(va, "full"), va[score].to_numpy(), tau1)
        p_te, used_te = route(P(te, "crop32"), P(te, "full"), te[score].to_numpy(), tau1)
        X = lambda p, d, used: np.column_stack([p.max(1), d[score].to_numpy(), used.astype(float)])  # noqa: E731
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        lr.fit(X(p_va, va, used_va), (p_va.argmax(1) == va.y).astype(int))
        te["routed_pred"] = p_te.argmax(1)
        te["routed_conf"] = p_te.max(1)
        te["combined_conf"] = lr.predict_proba(X(p_te, te, used_te))[:, 1]
        te["used_crop"] = used_te
        crop_ok, full_ok = P(te, "crop32").argmax(1) == te.y, P(te, "full").argmax(1) == te.y
        use_crop_oracle = crop_ok | ~full_ok
        p_or = np.where(use_crop_oracle[:, None], P(te, "crop32"), P(te, "full"))
        te["oracle_pred"], te["oracle_conf"] = p_or.argmax(1), p_or.max(1)
        conf = {}
        for alpha in (0.05, 0.10):
            for name, pv, pt in (("router", p_va, p_te), ("crop32", P(va, "crop32"), P(te, "crop32")),
                                 ("full", P(va, "full"), P(te, "full"))):
                q = conformal_threshold(pv, va.y.to_numpy(), alpha)
                sets = prediction_sets(pt, q)
                te[f"cp{alpha}_{name}_size"] = sets.sum(1)
                te[f"cp{alpha}_{name}_covered"] = sets[np.arange(len(te)), te.y.to_numpy()]
                conf[f"{name}@{alpha}"] = q
        fitted[k] = {"tau1": tau1, "frac_crop_val": float(used_va.mean()), "temperatures": folds[k]["temps"], "conformal_q": conf}
        tests.append(te)
    te = pd.concat(tests, ignore_index=True)
    y = te.y.to_numpy()
    pids = meta.loc[te.row, "pid"].to_numpy()

    policies = {
        "full+MSP": (P(te, "full").argmax(1), P(te, "full").max(1)),
        "masked+MSP": (P(te, "masked").argmax(1), P(te, "masked").max(1)),
        "crop32+MSP": (P(te, "crop32").argmax(1), P(te, "crop32").max(1)),
        "crop32+seg-unc": (P(te, "crop32").argmax(1), -te[score].to_numpy()),
        "router": (te.routed_pred.to_numpy(), te.routed_conf.to_numpy()),
        "router+combined": (te.routed_pred.to_numpy(), te.combined_conf.to_numpy()),
        "oracle-router": (te.oracle_pred.to_numpy(), te.oracle_conf.to_numpy()),
    }

    def stat(idx: np.ndarray) -> dict[str, float]:
        out = {}
        for name, (pred, c) in policies.items():
            ok = pred[idx] == y[idx]
            cm = clsm.confusion(y[idx], pred[idx])
            out[f"{name}|bal_acc@100"] = float(clsm.balanced_accuracy_c(cm))
            out[f"{name}|aurc"] = sel.aurc(c[idx], ok)
            out[f"{name}|augrc"] = sel.augrc(c[idx], ok)
            out[f"{name}|e_aurc"] = sel.e_aurc(c[idx], ok)
            for cov in COVERAGES:
                out[f"{name}|acc@{int(cov * 100)}"] = 1 - sel.risk_at_coverage(c[idx], ok, cov)
        for base in ("crop32+MSP", "full+MSP"):
            for m in ("bal_acc@100", "aurc", "augrc"):
                for name in ("router", "router+combined"):
                    out[f"{name} - {base}|{m}"] = out[f"{name}|{m}"] - out[f"{base}|{m}"]
        return out

    boot = paired_stat_bootstrap(pids, stat, n_boot=args.n_boot)

    # ---- conformal summary --------------------------------------------------------------------
    cp = {}
    for alpha in (0.05, 0.10):
        for name in ("router", "crop32", "full"):
            size = te[f"cp{alpha}_{name}_size"].to_numpy()
            cov = te[f"cp{alpha}_{name}_covered"].to_numpy()
            single = size == 1
            cp[f"{name}@alpha={alpha}"] = {
                "coverage": float(cov.mean()),
                "coverage_per_class": {c: float(cov[y == i].mean()) for i, c in enumerate(("meningioma", "glioma", "pituitary"))},
                "mean_set_size": float(size.mean()), "frac_singleton": float(single.mean()),
                "frac_empty": float((size == 0).mean()), "referral_rate": float((~single).mean()),
                "acc_on_singletons": float((te.loc[single, f"cp{alpha}_{name}_covered"]).mean()),
            }

    results = {"score": score, "val_auroc_crop32_error": val_auc, "fitted_per_fold": fitted,
               "frac_crop_test": float(te.used_crop.mean()), "policies": boot, "conformal": cp,
               "n_test": int(len(te)), "n_patients": int(len(np.unique(pids)))}
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    # ---- figure: risk-coverage curves --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    for name, col in POLICY_COLORS.items():
        pred, c = policies[name]
        cov, risk = sel.risk_coverage(c, pred == y)
        ax.plot(cov, risk, color=col, lw=1.5, ls="--" if name == "oracle-router" else "-", label=name)
    ax.set_xlabel("coverage (fraction of slices not referred)")
    ax.set_ylabel("selective error")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", bbox_inches="tight")

    print(f"tau1 per fold: {[round(f['tau1'], 4) for f in fitted.values()]}, crop used on test: {te.used_crop.mean():.3f}")
    for name in policies:
        b, a, g = boot[f"{name}|bal_acc@100"], boot[f"{name}|aurc"], boot[f"{name}|augrc"]
        print(f"{name:18s} bal.acc {b['estimate']:.3f} [{b['ci_low']:.3f},{b['ci_high']:.3f}]  AURC {a['estimate']:.4f} "
              f"[{a['ci_low']:.4f},{a['ci_high']:.4f}]  AUGRC {g['estimate']:.4f}  acc@90 {boot[f'{name}|acc@90']['estimate']:.3f}")
    for k, v in boot.items():
        if " - " in k:
            print(f"{k:40s} {v['estimate']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")
    for k, v in cp.items():
        print(f"CP {k:22s} cov {v['coverage']:.3f} size {v['mean_set_size']:.2f} referral {v['referral_rate']:.3f} "
              f"acc|singleton {v['acc_on_singletons']:.3f}")


if __name__ == "__main__":
    main()
