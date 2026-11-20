"""Export the M2 demo's policy parameters and example slices for one fold.

Parameters (temperatures, tau1, fill value, conformal thresholds) are fitted on the fold's
**val** frame with ``policy_eval.fit_params`` and the EXP-006 frozen score, exactly as in
EXP-006, and checked against the tau1 stored in EXP-006 ``results.json``.
Examples: a few test-fold slices (never seen by that fold's models), chosen by fixed rules
from EXP-006 ``test_routed.csv``: per class the first correctly classified slice routed to
the crop; the first slice routed to the full image; the first referred slice (alpha 0.1).

Usage:
    .venv/Scripts/python scripts/export_demo_params.py --config configs/demo.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mediscan import policy_eval as pe
from mediscan.demo import CLASS_NAMES
from mediscan.utils.repro import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    k = int(cfg["fold"])
    exp006 = json.loads(Path(cfg["exp006_results"]).read_text(encoding="utf-8"))
    score = exp006["score"]
    frames = pe.load_fold_frames(k, Path("runs"), [score.split(":")[0]])
    p = pe.fit_params(frames["val"], score)
    stored = float(exp006["fitted_per_fold"][str(k)]["tau1"])
    if not (p["tau1"] == stored or math.isclose(p["tau1"], stored, rel_tol=1e-9)):
        raise SystemExit(f"tau1 {p['tau1']} differs from EXP-006 {stored}")
    out = {"fold": k, "score": score, "fill": p["fill"], "tau1": p["tau1"],
           "temps": {v: float(t) for v, t in p["temps"].items()}, "q": {kk: float(v) for kk, v in p["q"].items()},
           "frac_crop_val": p["frac_crop_val"], "class_names": list(CLASS_NAMES),
           "ensemble_runs": cfg["ensemble_runs"], "source": "EXP-006 frozen protocol, fitted on val fold "
           f"{(k + 1) % 5} by scripts/export_demo_params.py"}
    Path(cfg["params_out"]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"params -> {cfg['params_out']}: tau1 {p['tau1']:.4f}, temps {out['temps']}, val crop share {p['frac_crop_val']:.2f}")

    te = pd.read_csv(cfg["routed_csv"])
    te = te[te.fold == k].sort_values("row")
    picks: list[tuple[int, str]] = []
    for c, name in enumerate(CLASS_NAMES):
        r = te[(te.y == c) & te.used_crop & (te.pred_router == te.y)].row
        if len(r):
            picks.append((int(r.iloc[0]), f"{name}_crop"))
    r = te[~te.used_crop].row
    if len(r):
        picks.append((int(r.iloc[0]), f"{CLASS_NAMES[int(te.loc[te.row == r.iloc[0], 'y'].iloc[0])]}_full"))
    r = te[te["cp0.1_router_size"] != 1].row
    if len(r):
        picks.append((int(r.iloc[0]), f"{CLASS_NAMES[int(te.loc[te.row == r.iloc[0], 'y'].iloc[0])]}_referred"))
    images = np.load("data/processed/figshare_256/images.npy", mmap_mode="r")
    ex = Path(cfg["examples_dir"])
    ex.mkdir(parents=True, exist_ok=True)
    for row, tag in picks:
        cv2.imwrite(str(ex / f"fold{k}_row{row}_{tag}.png"), np.asarray(images[row]))
    print(f"{len(picks)} examples -> {ex}: {[t for _, t in picks]}")


if __name__ == "__main__":
    main()
