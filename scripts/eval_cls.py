"""Aggregate EXP-003 runs: pooled out-of-fold test metrics, temperature scaling, patient-bootstrap CIs.

A "system" is one run evaluated with one mask source (``gt`` or ``pred``; full-image
runs have only ``gt``, which means "no mask used"). Per fold, the temperature is
fitted on that fold's validation logits of the same source, then applied to its
test logits (validation only; the test fold is never used for fitting).

Usage:
    .venv/Scripts/python scripts/eval_cls.py --runs exp003a_simplecnn_old exp003_r18_full ... \
        --out project-log/experiments/EXP-003-classification/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mediscan.calibration import fit_temperature
from mediscan.data.figshare import CLASS_NAMES
from mediscan.metrics import cls as clsm
from mediscan.stats import confusion_bootstrap, per_patient_confusion


def load_system(run_dir: Path, source: str) -> dict | None:
    """Pooled test logits (raw and temperature-scaled) for one run and mask source."""
    rows, labels, raw, scaled, temps, summaries = [], [], [], [], [], []
    for k in range(5):
        d = run_dir / f"fold{k}"
        if not (d / "summary.json").exists():
            raise SystemExit(f"missing {d}/summary.json")
        if not (d / f"logits_test_{source}.npz").exists():
            return None
        val, test = np.load(d / f"logits_val_{source}.npz"), np.load(d / f"logits_test_{source}.npz")
        t = fit_temperature(val["logits"], val["labels"])
        rows.append(test["rows"]); labels.append(test["labels"]); raw.append(test["logits"])
        scaled.append(test["logits"] / t); temps.append(t)
        summaries.append(json.loads((d / "summary.json").read_text(encoding="utf-8")))
    return {"rows": np.concatenate(rows), "y": np.concatenate(labels), "raw": np.concatenate(raw),
            "ts": np.concatenate(scaled), "temps": temps, "summaries": summaries}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--processed", default="data/processed/figshare_256")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    meta = pd.read_csv(Path(args.processed) / "metadata.csv", dtype={"pid": str})
    names = list(CLASS_NAMES)
    systems: dict[str, dict] = {}
    for run in args.runs:
        for source in ("gt", "pred"):
            s = load_system(Path(args.runs_root) / run, source)
            if s is not None:
                systems[f"{run}:{source}"] = s

    results: dict = {"systems": {}, "ci": {}, "paired": {}}
    confs: dict[str, np.ndarray] = {}
    ref_rows: np.ndarray | None = None
    for key, s in systems.items():
        order = np.argsort(s["rows"])
        for f in ("rows", "y", "raw", "ts"):
            s[f] = s[f][order]
        if ref_rows is None:
            ref_rows = s["rows"]
        assert np.array_equal(s["rows"], ref_rows), "systems cover different slices"
        pids = meta.loc[s["rows"], "pid"].to_numpy()
        _, confs[key] = per_patient_confusion(pids, s["y"], s["raw"].argmax(1), len(names))
        r = {"n_slices": int(len(s["rows"])), "n_patients": int(len(np.unique(pids)))}
        r["raw"] = clsm.all_metrics(s["raw"], s["y"], names)
        ts = clsm.all_metrics(s["ts"], s["y"], names)
        r["temperature_scaled"] = {k: ts[k] for k in ("ece", "brier", "nll", "auroc_ovr")}  # argmax unchanged by TS
        r["temperatures"] = [round(t, 4) for t in s["temps"]]
        folds = meta.loc[s["rows"], "fold"].to_numpy()
        r["per_fold_balanced_accuracy"] = [
            round(float(clsm.balanced_accuracy_c(clsm.confusion(s["y"][folds == k], s["raw"][folds == k].argmax(1), len(names)))), 4)
            for k in range(5)
        ]
        sm = s["summaries"]
        r["training"] = {k: [x[k] for x in sm] for k in ("best_epoch", "epochs_run", "train_time_s", "peak_vram_mib")}
        if "pred_mask_empty_test" in sm[0]:
            r["pred_mask_empty_test"] = int(sum(x["pred_mask_empty_test"] for x in sm))
        results["systems"][key] = r

    for metric in ("balanced_accuracy", "accuracy", "macro_f1", "mcc"):
        out = confusion_bootstrap(confs, clsm.COUNT_METRICS[metric], n_boot=args.n_boot)
        results["ci"][metric] = out["single"]
        results["paired"][metric] = out["diff"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"{'system':42s} {'bal.acc [95% CI]':26s} {'acc':>6s} {'mF1':>6s} {'MCC':>6s} {'AUROC':>6s} {'ECE':>6s} {'ECE-TS':>6s}")
    for key, r in results["systems"].items():
        b = results["ci"]["balanced_accuracy"][key]
        print(f"{key:42s} {b['estimate']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]     "
              f"{r['raw']['accuracy']:6.3f} {r['raw']['macro_f1']:6.3f} {r['raw']['mcc']:6.3f} "
              f"{r['raw']['auroc_ovr']:6.3f} {r['raw']['ece']:6.3f} {r['temperature_scaled']['ece']:6.3f}")


if __name__ == "__main__":
    main()
