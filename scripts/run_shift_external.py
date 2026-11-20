"""M1.4: run the full pipeline under synthetic shift (EXP-008) or on the external set (EXP-009).

* ``--mode shift``: every test fold k (fold-k models) with each corruption x severity
  (``mediscan.corrupt``); also severity 0 (clean) as a consistency check against EXP-003/005.
* ``--mode external``: the clean BDNeuro-MRI subset (``audit == "clean"``) through each of
  the 5 fold model sets (none of them saw any BDNeuro image).
Only the uncertainty method(s) in ``--methods`` are computed (default: det + the method
frozen in EXP-006), to keep GPU time bounded.

Usage:
    .venv/Scripts/python scripts/run_shift_external.py --mode shift --methods det mc
    .venv/Scripts/python scripts/run_shift_external.py --mode external --methods det mc
Writes ``runs/exp008_shift/<kind>_s<sev>_fold<k>.csv`` or ``runs/exp009_external/fold<k>.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mediscan.corrupt import KINDS, corrupt
from mediscan.data.datasets import FigshareArrays
from mediscan.data.splits import load_split
from mediscan.pipeline import run_fold
from mediscan.utils.repro import create_run_dir, load_config, seed_everything

ENSEMBLE = ["exp002c_unet_r34", "exp002c_unet_r34_s1", "exp002c_unet_r34_s2"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["shift", "external"], required=True)
    p.add_argument("--methods", nargs="+", default=["det", "mc"])
    p.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--kinds", nargs="*", default=list(KINDS))
    p.add_argument("--severities", type=int, nargs="*", default=[0, 1, 2, 3])
    args = p.parse_args()
    runs = Path("runs")
    seed_everything(0, deterministic=True)
    name = "exp008_shift" if args.mode == "shift" else "exp009_external"
    create_run_dir(runs, name, vars(args), exist_ok=True)
    out_root = runs / name

    if args.mode == "external":
        meta = pd.read_csv("data/processed/bdneuro_256/metadata.csv")
        images = np.load("data/processed/bdneuro_256/images.npy")
        keep = meta.index[meta.audit == "clean"].to_numpy()
        for k in args.folds:
            out = out_root / f"fold{k}.csv"
            if out.exists():
                continue
            t0 = time.time()
            df = run_fold(images[keep], k, runs, args.methods, ENSEMBLE)
            df.insert(0, "ext_row", keep[df.idx.to_numpy()])
            df.insert(1, "y", meta.class_index.to_numpy()[keep][df.idx.to_numpy()])
            df.to_csv(out, index=False)
            print(f"external fold {k}: {len(df)} images in {time.time() - t0:.0f}s", flush=True)
        return

    arrays = FigshareArrays.load("data/processed/figshare_256")
    for k in args.folds:
        split = load_split(Path("data/splits") / f"figshare_fold{k}.json", arrays.meta)
        rows = np.array(split.test)
        for kind in args.kinds:
            for sev in args.severities:
                if sev == 0 and kind != args.kinds[0]:
                    continue  # the clean condition is run once per fold
                tag = "clean_s0" if sev == 0 else f"{kind}_s{sev}"
                out = out_root / f"{tag}_fold{k}.csv"
                if out.exists():
                    continue
                t0 = time.time()
                if sev == 0:
                    imgs = arrays.images[rows]
                else:
                    ki = KINDS.index(kind)
                    imgs = np.stack([corrupt(arrays.images[r], kind, sev, np.random.default_rng([0, int(r), ki, sev]))
                                     for r in rows])
                df = run_fold(imgs, k, runs, args.methods, ENSEMBLE, gt_masks=arrays.masks[rows])
                df.insert(0, "row", rows[df.idx.to_numpy()])
                df.insert(1, "y", arrays.meta.loc[rows, "class_index"].to_numpy()[df.idx.to_numpy()])
                df.to_csv(out, index=False)
                print(f"fold {k} {tag}: {len(df)} images in {time.time() - t0:.0f}s, Dice {df.dice_lcc.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
