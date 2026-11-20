"""EXP-008 (synthetic shift) and EXP-009 (external BDNeuro-MRI) analysis of the frozen routing policies.

All policy parameters are fitted per fold on the **clean in-distribution val** frames
(``mediscan.policy_eval.fit_params``) with the uncertainty score frozen in EXP-006, and are
then applied unchanged to the shifted test folds and to the external images.

H-B3 (plan): the benefit of routing/referral grows with shift severity and is larger on
the external set than in-distribution.

Usage:
    .venv/Scripts/python scripts/eval_shift_external.py --exp006 project-log/experiments/EXP-006-router/results.json \
        --out-shift project-log/experiments/EXP-008-shift --out-external project-log/experiments/EXP-009-external
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

from mediscan import policy_eval as pe  # noqa: E402
from mediscan.corrupt import KINDS  # noqa: E402

KEY = ["router - crop32+MSP|bal_acc@100", "router - full+MSP|bal_acc@100", "router - crop32+MSP|aurc",
       "router - full+MSP|aurc", "router+combined - crop32+MSP|aurc"]
SINGLE = ["full+MSP", "crop32+MSP", "masked+MSP", "router", "router+combined", "oracle-router"]
KIND_COLORS = {"noise": "#2a78d6", "blur": "#eb6834", "gamma": "#1baf7a", "bias": "#eda100"}


def summarize(te: pd.DataFrame, ids: np.ndarray, n_boot: int) -> dict:
    boot = pe.bootstrap_policies(te, ids, n_boot)
    out = {"n": int(len(te)), "frac_crop": float(te.used_crop.mean())}
    out["single"] = {p: {m: boot[f"{p}|{m}"] for m in ("bal_acc@100", "aurc", "augrc", "acc@90")} for p in SINGLE}
    out["diff"] = {k: boot[k] for k in KEY}
    out["conformal"] = pe.conformal_summary(te)
    if "dice_lcc" in te:
        out["seg_dice_mean"] = float(te.dice_lcc.mean())
        out["seg_frac_dice_below_0.3"] = float((te.dice_lcc < 0.3).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp006", required=True)
    ap.add_argument("--out-shift", required=True)
    ap.add_argument("--out-external", required=True)
    ap.add_argument("--methods", nargs="*", default=["det", "mc", "tta", "ens"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    runs = Path("runs")
    score = json.loads(Path(args.exp006).read_text(encoding="utf-8"))["score"]
    meta = pd.read_csv("data/processed/figshare_256/metadata.csv", dtype={"pid": str})
    params = {}
    for k in range(5):
        frames = pe.load_fold_frames(k, runs, args.methods)
        params[k] = pe.fit_params(frames["val"], score)
    print(f"frozen score {score}; tau1 {[round(params[k]['tau1'], 4) for k in range(5)]}")

    # ---- EXP-008: synthetic shift ------------------------------------------------------------
    out_s = Path(args.out_shift)
    out_s.mkdir(parents=True, exist_ok=True)
    shift: dict = {}
    conds = [("clean", 0)] + [(kd, s) for kd in KINDS for s in (1, 2, 3)]
    for kind, sev in conds:
        tag = "clean_s0" if sev == 0 else f"{kind}_s{sev}"
        files = [runs / "exp008_shift" / f"{tag}_fold{k}.csv" for k in range(5)]
        if not all(f.exists() for f in files):
            print(f"{tag}: missing, skipped")
            continue
        te = pd.concat([pe.apply_params(params[k], pd.read_csv(f)) for k, f in enumerate(files)], ignore_index=True)
        shift[tag] = summarize(te, meta.loc[te.row, "pid"].to_numpy(), args.n_boot)
        d = shift[tag]["diff"]
        print(f"{tag:9s} Dice {shift[tag]['seg_dice_mean']:.3f}  crop {shift[tag]['frac_crop']:.2f}  "
              f"full {shift[tag]['single']['full+MSP']['bal_acc@100']['estimate']:.3f}  "
              f"crop32 {shift[tag]['single']['crop32+MSP']['bal_acc@100']['estimate']:.3f}  "
              f"router {shift[tag]['single']['router']['bal_acc@100']['estimate']:.3f}  "
              f"oracle {shift[tag]['single']['oracle-router']['bal_acc@100']['estimate']:.3f}  "
              f"R-crop {d[KEY[0]]['estimate']:+.3f} [{d[KEY[0]]['ci_low']:+.3f},{d[KEY[0]]['ci_high']:+.3f}]  "
              f"R-full {d[KEY[1]]['estimate']:+.3f} [{d[KEY[1]]['ci_low']:+.3f},{d[KEY[1]]['ci_high']:+.3f}]", flush=True)
    (out_s / "results.json").write_text(json.dumps({"score": score, "conditions": shift}, indent=1), encoding="utf-8")

    if shift:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), dpi=200)
        series = [("full+MSP", "full image"), ("crop32+MSP", "crop 32 px"), ("router", "router")]
        for ax, (pol, label) in zip(axes, series):
            for kind, col in KIND_COLORS.items():
                xs, ys, lo, hi = [], [], [], []
                for sev in (0, 1, 2, 3):
                    tag = "clean_s0" if sev == 0 else f"{kind}_s{sev}"
                    if tag in shift:
                        v = shift[tag]["single"][pol]["bal_acc@100"]
                        xs.append(sev); ys.append(v["estimate"]); lo.append(v["ci_low"]); hi.append(v["ci_high"])
                ax.plot(xs, ys, "-o", color=col, lw=1.5, ms=4, label=kind)
                ax.fill_between(xs, lo, hi, color=col, alpha=0.10, lw=0)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xlabel("corruption severity (0 = clean)")
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("balanced accuracy (test, 5 folds)")
        lo_all = min(a.get_ylim()[0] for a in axes)
        for a in axes:
            a.set_ylim(lo_all, 1.0)
        axes[0].legend(frameon=False, fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(out_s / "accuracy_vs_severity.png", bbox_inches="tight")

    # ---- EXP-009: external (clean BDNeuro subset), one result per fold model set ---------------
    out_e = Path(args.out_external)
    out_e.mkdir(parents=True, exist_ok=True)
    ext: dict = {}
    for k in range(5):
        f = runs / "exp009_external" / f"fold{k}.csv"
        if not f.exists():
            continue
        te = pe.apply_params(params[k], pd.read_csv(f))
        ext[f"fold{k}"] = summarize(te, te.ext_row.to_numpy(), args.n_boot)  # no patient IDs: image-level bootstrap
    if ext:
        agg = {}
        for p in SINGLE:
            for m in ("bal_acc@100", "aurc", "acc@90"):
                v = [ext[f]["single"][p][m]["estimate"] for f in ext]
                agg[f"{p}|{m}"] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
        for kk in KEY:
            v = [ext[f]["diff"][kk]["estimate"] for f in ext]
            sig = [ext[f]["diff"][kk]["ci_low"] > 0 or ext[f]["diff"][kk]["ci_high"] < 0 for f in ext]
            agg[kk] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n_folds_ci_excludes_0": int(sum(sig))}
        ext["aggregate_over_fold_models"] = agg
        for kk, v in agg.items():
            print(f"external {kk:40s} mean {v['mean']:+.4f}  range [{v['min']:+.4f}, {v['max']:+.4f}]"
                  + (f"  CI excl. 0 in {v['n_folds_ci_excludes_0']}/5" if "n_folds_ci_excludes_0" in v else ""))
    (out_e / "results.json").write_text(json.dumps({"score": score, "external": ext}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
