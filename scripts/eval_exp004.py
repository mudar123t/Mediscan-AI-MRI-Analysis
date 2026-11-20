"""Analyse EXP-004 (R3): balanced accuracy vs mask quality x context, plus the real predicted masks.

Part A (controlled): per view, failure type and target Dice, balanced accuracy with a
patient-bootstrap CI and the paired difference to the mask-free full-image model
(EXP-003 ``exp003_r18_full``) on the same test slices.

Part B (real): EXP-003 predicted-mask logits, slices binned by the actual Dice of the
predicted mask they used (EXP-002c, largest component). Per bin: each view vs full image.

Usage:
    .venv/Scripts/python scripts/eval_exp004.py --config configs/exp004_mask_quality.yaml \
        --out project-log/experiments/EXP-004-mask-quality
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

from mediscan.metrics import cls as clsm  # noqa: E402
from mediscan.stats import confusion_bootstrap, per_patient_confusion  # noqa: E402
from mediscan.utils.repro import load_config  # noqa: E402

DICE_BINS = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0001]
VIEW_LABEL = {"exp003_r18_masked": "masked ROI", "exp003_r18_crop0": "crop 0 px", "exp003_r18_crop8": "crop 8 px",
              "exp003_r18_crop16": "crop 16 px", "exp003_r18_crop32": "crop 32 px"}
# Ordinal one-hue ramp (light -> dark = less -> more context); reference in neutral grey.
RAMP = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GREY = "#6b6a65"


def compare(pids: np.ndarray, y: np.ndarray, pred_ref: np.ndarray, pred_view: np.ndarray, n_boot: int) -> dict:
    _, c_ref = per_patient_confusion(pids, y, pred_ref)
    _, c_v = per_patient_confusion(pids, y, pred_view)
    out = confusion_bootstrap({"full": c_ref, "view": c_v}, clsm.balanced_accuracy_c, n_boot=n_boot)
    return {"view": out["single"]["view"], "full": out["single"]["full"], "view_minus_full": out["diff"]["view - full"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    cfg = load_config(args.config)
    runs = Path("runs")
    meta = pd.read_csv("data/processed/figshare_256/metadata.csv", dtype={"pid": str})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Part A: controlled degradations -------------------------------------------------
    parts = [np.load(runs / cfg["run_name"] / f"fold{k}.npz") for k in cfg["folds"]]
    rows = np.concatenate([z["rows"] for z in parts])
    y = np.concatenate([z["labels"] for z in parts])
    achieved = np.concatenate([z["achieved_dice"] for z in parts], axis=1)
    logits = np.concatenate([z["logits"] for z in parts], axis=2)
    kinds, targets = list(parts[0]["kinds"]), parts[0]["targets"].tolist()
    view_runs = [str(v) for v in parts[0]["view_runs"]]
    ref = {}
    for k in cfg["folds"]:
        z = np.load(runs / cfg["reference_run"] / f"fold{k}" / "logits_test_gt.npz")
        ref.update(dict(zip(z["rows"].tolist(), z["logits"].argmax(1))))
    pred_full = np.array([ref[r] for r in rows])
    pids = meta.loc[rows, "pid"].to_numpy()

    A: list[dict] = []
    for v, run in enumerate(view_runs):
        for c, (kind, target) in enumerate(zip(kinds, targets)):
            r = compare(pids, y, pred_full, logits[v, c].argmax(1), args.n_boot)
            A.append({"view": run, "kind": str(kind), "target": target, "achieved_dice_mean": float(achieved[c].mean()),
                      "frac_within_0.05": float(np.mean(np.abs(achieved[c] - target) < 0.05)),
                      "empty_masks": None, **{f"bal_acc_{k}": r["view"][k] for k in ("estimate", "ci_low", "ci_high")},
                      **{f"diff_vs_full_{k}": r["view_minus_full"][k] for k in ("diff", "ci_low", "ci_high", "p_two_sided")}})
    full_ci = compare(pids, y, pred_full, pred_full, args.n_boot)["full"]
    dfA = pd.DataFrame(A).drop(columns="empty_masks")
    dfA.to_csv(out_dir / "controlled.csv", index=False)

    # ---- Part B: real predicted masks, binned by their actual Dice ------------------------
    seg = pd.concat([pd.read_csv(runs / "exp002c_unet_r34" / f"fold{k}" / "metrics_test.csv") for k in cfg["folds"]])
    seg_dice = dict(zip(seg["row"], seg["dice_lcc"]))
    B: list[dict] = []
    for run in view_runs:
        lg, rr = [], []
        for k in cfg["folds"]:
            z = np.load(runs / run / f"fold{k}" / "logits_test_pred.npz")
            lg.append(z["logits"]); rr.append(z["rows"])
        lg, rr = np.concatenate(lg), np.concatenate(rr)
        order = np.argsort(rr)
        lg, rr = lg[order], rr[order]
        yb = meta.loc[rr, "class_index"].to_numpy()
        pf = np.array([ref[r] for r in rr])
        d = np.array([seg_dice[r] for r in rr])
        pb = meta.loc[rr, "pid"].to_numpy()
        for lo, hi in zip(DICE_BINS[:-1], DICE_BINS[1:]):
            sel = (d >= lo) & (d < hi)
            r = compare(pb[sel], yb[sel], pf[sel], lg[sel].argmax(1), args.n_boot)
            B.append({"view": run, "dice_bin": f"[{lo:.2f}, {min(hi, 1.0):.2f}]", "n_slices": int(sel.sum()),
                      "n_patients": int(len(np.unique(pb[sel]))),
                      "acc_view": float(np.mean(lg[sel].argmax(1) == yb[sel])), "acc_full": float(np.mean(pf[sel] == yb[sel])),
                      **{f"bal_acc_view_{k}": r["view"][k] for k in ("estimate", "ci_low", "ci_high")},
                      "bal_acc_full": r["full"]["estimate"],
                      **{f"diff_vs_full_{k}": r["view_minus_full"][k] for k in ("diff", "ci_low", "ci_high", "p_two_sided")}})
    dfB = pd.DataFrame(B)
    dfB.to_csv(out_dir / "predicted_by_dice.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps({"full_reference": full_ci, "controlled": A, "predicted_by_dice": B},
                                                    indent=1), encoding="utf-8")

    # ---- Figure: one panel per failure type + one for real predicted masks ---------------
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9), sharey=True, dpi=200)
    real = [k for k in dict.fromkeys(kinds) if k != "none"]
    for ax, kind in zip(axes, real):
        ax.axhspan(full_ci["ci_low"], full_ci["ci_high"], color=GREY, alpha=0.12, lw=0)
        ax.axhline(full_ci["estimate"], color=GREY, lw=1.5, ls="--", label="full image (no mask)")
        for col, run in zip(RAMP, view_runs):
            sub = dfA[(dfA.view == run) & ((dfA.kind == kind) | (dfA.kind == "none"))].sort_values("achieved_dice_mean")
            ax.plot(sub.achieved_dice_mean, sub.bal_acc_estimate, "-o", color=col, lw=1.5, ms=4, label=VIEW_LABEL[run])
            ax.fill_between(sub.achieved_dice_mean, sub.bal_acc_ci_low, sub.bal_acc_ci_high, color=col, alpha=0.10, lw=0)
        ax.set_title(f"GT mask degraded by: {kind}", fontsize=10)
        ax.set_xlabel("mask Dice vs GT (achieved mean)")
        ax.invert_xaxis()
    ax = axes[-1]
    centers = [(lo + min(hi, 1.0)) / 2 for lo, hi in zip(DICE_BINS[:-1], DICE_BINS[1:])]
    for col, run in zip(RAMP, view_runs):
        sub = dfB[dfB.view == run]
        ax.plot(centers, sub.bal_acc_view_estimate, "-o", color=col, lw=1.5, ms=4, label=VIEW_LABEL[run])
    ax.plot(centers, dfB[dfB.view == view_runs[0]].bal_acc_full, "--o", color=GREY, lw=1.5, ms=4, label="full image (no mask)")
    ax.set_title("Real predicted masks, by Dice bin", fontsize=10)
    ax.set_xlabel("predicted-mask Dice (bin centre)")
    ax.invert_xaxis()
    axes[0].set_ylabel("balanced accuracy (test, 5 folds)")
    for a in axes:
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7.5, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_mask_quality.png", bbox_inches="tight")

    # ---- Console summary -------------------------------------------------------------------
    print(f"full image reference: {full_ci['estimate']:.3f} [{full_ci['ci_low']:.3f}, {full_ci['ci_high']:.3f}]")
    piv = dfA.pivot_table(index=["kind", "target"], columns="view", values="bal_acc_estimate")[view_runs]
    print(piv.round(3).to_string())
    print(dfB[["view", "dice_bin", "n_slices", "n_patients", "bal_acc_view_estimate", "bal_acc_full", "diff_vs_full_diff",
               "diff_vs_full_ci_low", "diff_vs_full_ci_high"]].round(3).to_string())


if __name__ == "__main__":
    main()
