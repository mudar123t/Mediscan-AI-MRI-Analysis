"""Analyse EXP-005: does segmentation uncertainty predict segmentation failure and classifier error?

Targets on the pooled test folds:
* ``seg_fail``: EXP-002c predicted mask (largest component) has Dice < 0.7;
* ``cls_err_crop32``: the crop-32 classifier on predicted masks (EXP-003) is wrong;
* ``cls_err_full``: the full-image classifier (EXP-003) is wrong.
Per (method, score): AUROC with a patient-bootstrap CI, and Spearman correlation with Dice.
Scores are oriented so that higher = more suspicious (``maxprob_ring`` is negated).
Missing values (empty reference mask) get the most suspicious value.

"Adds to MSP" (H for EXP-005): per fold, a logistic regression on the fold's **val** slices
predicts crop-32 classifier error from [MSP] or [MSP, score]; it is applied to the fold's
test slices. Test AUROCs of the two are compared with a paired patient bootstrap.

Usage:
    .venv/Scripts/python scripts/eval_exp005.py --config configs/exp005_uncertainty.yaml \
        --out project-log/experiments/EXP-005-uncertainty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mediscan.metrics.cls import softmax
from mediscan.stats import paired_stat_bootstrap
from mediscan.uncertainty import SCORE_NAMES
from mediscan.utils.repro import load_config

CLS_RUN, FULL_RUN = "exp003_r18_crop32", "exp003_r18_full"


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def load_table(cfg: dict, runs: Path, meta: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """One row per (row, part): scores of every method + targets + classifier MSP."""
    frames = []
    for k in cfg["folds"]:
        df = None
        for m in methods:
            s = pd.read_csv(runs / cfg["run_name"] / f"scores_fold{k}_{m}.csv")
            s = s.rename(columns={c: f"{m}:{c}" for c in SCORE_NAMES})
            df = s if df is None else df.merge(s, on=["row", "part"])
        df["fold"] = k
        seg = pd.concat([pd.read_csv(runs / cfg["seg_run"] / f"fold{k}" / f"metrics_{p}.csv").assign(part=p) for p in ("val", "test")])
        df = df.merge(seg[["row", "part", "dice_lcc"]], on=["row", "part"])
        for run, tag in ((CLS_RUN, "crop32"), (FULL_RUN, "full")):
            src = "pred" if run == CLS_RUN else "gt"
            for part in ("val", "test"):
                z = np.load(runs / run / f"fold{k}" / f"logits_{part}_{src}.npz")
                pr = softmax(z["logits"])
                t = pd.DataFrame({"row": z["rows"], "part": part, f"err_{tag}": (pr.argmax(1) != z["labels"]).astype(int),
                                  f"msp_{tag}": pr.max(1)})
                df = df.merge(t, on=["row", "part"], how="left", suffixes=("", "_dup"))
                for c in (f"err_{tag}", f"msp_{tag}"):
                    if f"{c}_dup" in df:
                        df[c] = df[c].fillna(df.pop(f"{c}_dup"))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["pid"] = meta.loc[out["row"], "pid"].to_numpy()
    out["seg_fail"] = (out["dice_lcc"] < 0.7).astype(int)
    for m in methods:
        out[f"{m}:maxprob_ring"] = -out[f"{m}:maxprob_ring"]  # higher = more suspicious
    score_cols = [c for c in out if ":" in c]
    for c in score_cols:
        out[c] = out[c].fillna(out[c].max())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="*", default=["det", "mc", "tta", "ens"])
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    cfg = load_config(args.config)
    runs = Path("runs")
    meta = pd.read_csv("data/processed/figshare_256/metadata.csv", dtype={"pid": str})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(cfg, runs, meta, args.methods)
    te = df[df.part == "test"].reset_index(drop=True)
    score_cols = [c for c in df if ":" in c]
    targets = {"seg_fail": te.seg_fail.to_numpy(), "cls_err_crop32": te.err_crop32.to_numpy().astype(int),
               "cls_err_full": te.err_full.to_numpy().astype(int)}
    S = {c: te[c].to_numpy() for c in score_cols}
    S["msp_crop32 (classifier)"] = -te.msp_crop32.to_numpy()
    S["msp_full (classifier)"] = -te.msp_full.to_numpy()
    # predicted-mask area: the "no uncertainty" baseline (small masks fail more often)
    area = []
    for k in cfg["folds"]:
        seg = pd.read_csv(runs / cfg["seg_run"] / f"fold{k}" / "metrics_test.csv")
        area.append(seg[["row", "pred_area"]])
    area = pd.concat(area).set_index("row")["pred_area"]
    S["mask area (small = suspicious)"] = -np.log1p(area.loc[te.row].to_numpy())

    pids = te.pid.to_numpy()
    res_auc: dict = {}
    for tname, y in targets.items():
        stat = lambda idx, y=y: {c: _auc(y[idx], s[idx]) for c, s in S.items()}  # noqa: E731
        res_auc[tname] = paired_stat_bootstrap(pids, stat, n_boot=args.n_boot)
    rho = {c: float(spearmanr(s, -te.dice_lcc.to_numpy()).statistic) for c, s in S.items()}

    # ---- Adds to MSP? per-fold logistic regression fitted on val, applied to test ----------
    combos = {"MSP only": ["msp_crop32"]}
    for c in score_cols:
        combos[f"MSP + {c}"] = ["msp_crop32", c]
    preds = {name: np.zeros(len(te)) for name in combos}
    for k in cfg["folds"]:
        va = df[(df.part == "val") & (df.fold == k)]
        mk = (te.fold == k).to_numpy()
        for name, cols in combos.items():
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            lr.fit(va[cols].to_numpy(), va.err_crop32.astype(int).to_numpy())
            preds[name][mk] = lr.predict_proba(te.loc[mk, cols].to_numpy())[:, 1]
    y = targets["cls_err_crop32"]

    def stat_combo(idx: np.ndarray) -> dict[str, float]:
        base = _auc(y[idx], preds["MSP only"][idx])
        out = {"MSP only": base}
        for name in combos:
            if name != "MSP only":
                a = _auc(y[idx], preds[name][idx])
                out[name] = a
                out[f"{name} minus MSP"] = a - base
        return out

    res_combo = paired_stat_bootstrap(pids, stat_combo, n_boot=args.n_boot)

    results = {"n_test": int(len(te)), "n_patients": int(te.pid.nunique()),
               "positives": {k: int(v.sum()) for k, v in targets.items()},
               "auroc": res_auc, "spearman_vs_neg_dice": rho, "adds_to_msp": res_combo}
    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    rows = []
    for c in S:
        r = {"score": c, "spearman_vs_-dice": rho[c]}
        for tname in targets:
            a = res_auc[tname][c]
            r[tname] = f"{a['estimate']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]"
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "auroc_table.csv", index=False)
    print("positives:", results["positives"], "of", results["n_test"])
    print(table.to_string(index=False))
    print("\nAdds to MSP (crop-32 classifier error, test AUROC):")
    for k, v in res_combo.items():
        print(f"  {k:45s} {v['estimate']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")


if __name__ == "__main__":
    main()
