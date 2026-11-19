"""Aggregate EXP-002 runs: pooled out-of-fold test metrics with patient-bootstrap CIs.

Usage:
    .venv/Scripts/python scripts/eval_seg.py --runs exp002a_unet_s1_old exp002b_unet_s1_new exp002c_unet_r34 \
        --out project-log/experiments/EXP-002-segmentation/results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from mediscan.stats import mean_ci, paired_patient_bootstrap

METRICS = ("dice", "iou", "hd95", "nsd", "dice_lcc", "hd95_lcc")


def load_run(run_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    frames, summaries = [], []
    for k in range(5):
        fold_dir = run_dir / f"fold{k}"
        if not (fold_dir / "summary.json").exists():
            raise SystemExit(f"missing {fold_dir}/summary.json")
        df = pd.read_csv(fold_dir / "metrics_test.csv", dtype={"pid": str})
        df["fold"] = k
        frames.append(df)
        summaries.append(json.loads((fold_dir / "summary.json").read_text(encoding="utf-8")))
    return pd.concat(frames, ignore_index=True), summaries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    results: dict = {"runs": {}, "paired": {}}
    pooled: dict[str, pd.DataFrame] = {}
    for name in args.runs:
        df, summaries = load_run(Path(args.runs_root) / name)
        pooled[name] = df
        r: dict = {"n_slices": len(df), "n_patients": int(df["pid"].nunique())}
        r["pooled"] = {m: mean_ci(df, m, n_boot=args.n_boot) for m in METRICS}
        r["patient_level_dice"] = float(df.groupby("pid")["dice"].mean().mean())
        r["frac_dice_below_0.5"] = float((df["dice"] < 0.5).mean())
        r["frac_empty_pred"] = float((df["pred_area"] == 0).mean())
        per_fold = df.groupby("fold")["dice"].mean()
        r["per_fold_dice"] = per_fold.round(4).tolist()
        r["fold_dice_mean_sd"] = [float(per_fold.mean()), float(per_fold.std(ddof=1))]
        r["per_class"] = {
            c: {m: mean_ci(g, m, n_boot=args.n_boot) for m in ("dice", "hd95")}
            for c, g in df.groupby("class_name")
        }
        r["training"] = {
            "best_epoch": [s["best_epoch"] for s in summaries],
            "epochs_run": [s["epochs_run"] for s in summaries],
            "train_time_s": [s["train_time_s"] for s in summaries],
            "peak_vram_mib": [s["peak_vram_mib"] for s in summaries],
        }
        results["runs"][name] = r

    for a, b in itertools.combinations(args.runs, 2):
        merged = pooled[a][["row", "pid", "dice", "hd95"]].merge(
            pooled[b][["row", "dice", "hd95"]], on="row", suffixes=("_a", "_b")
        )
        assert len(merged) == len(pooled[a]) == len(pooled[b]), "runs cover different slices"
        results["paired"][f"{b} - {a}"] = {
            m: paired_patient_bootstrap(merged, f"{m}_b", f"{m}_a", n_boot=args.n_boot) for m in ("dice", "hd95")
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    for name, r in results["runs"].items():
        d, h, n = r["pooled"]["dice"], r["pooled"]["hd95"], r["pooled"]["nsd"]
        print(f"{name}: Dice {d['estimate']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]  "
              f"HD95 {h['estimate']:.1f} px  NSD {n['estimate']:.3f}  folds {r['fold_dice_mean_sd'][0]:.3f}±{r['fold_dice_mean_sd'][1]:.3f}")
    for k, v in results["paired"].items():
        d = v["dice"]
        print(f"{k}: dDice {d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] p={d['p_two_sided']:.4f}")


if __name__ == "__main__":
    main()
