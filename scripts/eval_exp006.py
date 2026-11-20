"""EXP-006 (R1 routing) and EXP-007 (R7 conformal referral) on the pooled in-distribution test folds.

No training: EXP-003 view logits + EXP-005 uncertainty scores. Per fold, everything is
fitted on that fold's val slices (``mediscan.policy_eval``). The uncertainty score that
drives the router is chosen on **pooled val** data (AUROC for crop-32 classifier error),
never on test, and is then frozen for EXP-008/009.

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
from sklearn.metrics import roc_auc_score  # noqa: E402

from mediscan import policy_eval as pe  # noqa: E402
from mediscan.metrics import cls as clsm  # noqa: E402
from mediscan.metrics import selective as sel  # noqa: E402

POLICY_COLORS = {"full+MSP": "#6b6a65", "crop32+MSP": "#2a78d6", "router": "#eb6834", "router+combined": "#1baf7a",
                 "oracle-router": "#9a9893"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="*", default=["det", "mc", "tta", "ens"])
    ap.add_argument("--score", default=None, help="force the uncertainty score (default: best pooled-val AUROC)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    runs = Path("runs")
    meta = pd.read_csv("data/processed/figshare_256/metadata.csv", dtype={"pid": str})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {k: pe.load_fold_frames(k, runs, args.methods) for k in range(5)}

    # 1. choose the uncertainty score on pooled val (crop-32 error AUROC; each fold's own temperature is irrelevant to AUROC)
    val = pd.concat([frames[k]["val"] for k in range(5)], ignore_index=True)
    score_cols = [c for c in val if ":" in c]
    err = (clsm.softmax(pe.logits(val, "crop32")).argmax(1) != val.y).astype(int)
    val_auc = {c: float(roc_auc_score(err, val[c].fillna(val[c].max()))) for c in score_cols}
    score = args.score or max(val_auc, key=val_auc.get)
    print(f"uncertainty score: {score} (pooled val AUROC for crop-32 error {val_auc[score]:.3f})")

    # 2. fit per fold on val, apply to test
    tests, fitted = [], {}
    for k in range(5):
        params = pe.fit_params(frames[k]["val"], score)
        tests.append(pe.apply_params(params, frames[k]["test"]))
        fitted[k] = {kk: vv for kk, vv in params.items() if kk != "lr"}
    te = pd.concat(tests, ignore_index=True)
    pids = meta.loc[te.row, "pid"].to_numpy()
    boot = pe.bootstrap_policies(te, pids, args.n_boot)
    cp = pe.conformal_summary(te)

    # routing behaviour: when the router used the full view, how good was the mask?
    seg = pd.concat([pd.read_csv(runs / "exp002c_unet_r34" / f"fold{k}" / "metrics_test.csv") for k in range(5)])
    te = te.merge(seg[["row", "dice_lcc"]], on="row")
    routing = {"frac_crop_test": float(te.used_crop.mean()),
               "median_dice_when_crop": float(te.loc[te.used_crop, "dice_lcc"].median()),
               "median_dice_when_full": float(te.loc[~te.used_crop, "dice_lcc"].median()) if (~te.used_crop).any() else None}

    results = {"score": score, "val_auroc_crop32_error": val_auc, "fitted_per_fold": fitted, "routing": routing,
               "policies": boot, "conformal": cp, "n_test": int(len(te)), "n_patients": int(len(np.unique(pids)))}
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    te.drop(columns=[c for c in te if c.startswith("l_")]).to_csv(out_dir / "test_routed.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    pol = pe.policies(te)
    y = te.y.to_numpy()
    for name, col in POLICY_COLORS.items():
        pred, c = pol[name]
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

    print(f"tau1 per fold: {[round(f['tau1'], 4) for f in fitted.values()]}; routing: {routing}")
    for name in pol:
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
