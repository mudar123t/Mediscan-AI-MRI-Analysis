"""EXP-011 (R2): MedSAM second-opinion signals, pre-specified in project-log/experiments/EXP-011-medsam-second-opinion.md.

Modes (each output unit is skipped if it exists, so the run resumes after a crash):
* ``clean``    val + test slices of each fold, box from that fold's stored EXP-002c mask
               -> runs/exp011_medsam/clean_fold{k}.csv
* ``external`` the 3,030 clean BDNeuro images, boxes from each fold's U-Net
               -> runs/exp011_medsam/external_fold{k}.csv
* ``shift``    the 12 corrupted EXP-008 conditions (same RNG as EXP-008), test fold k
               -> runs/exp011_medsam/shift_<kind>_s<sev>_fold{k}.csv
               (check: U-Net Dice must equal the stored EXP-008 ``dice_lcc``)

Usage:
    .venv/Scripts/python scripts/run_exp011.py --mode clean|external|shift
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mediscan.corrupt import KINDS, corrupt
from mediscan.data.datasets import FigshareArrays, load_predicted_masks
from mediscan.data.splits import load_split
from mediscan.medsam import MedSAM
from mediscan.metrics.seg import dice
from mediscan.pipeline import FoldModels
from mediscan.utils.repro import create_run_dir, seed_everything

OUT = Path("runs") / "exp011_medsam"


def unet_refs(images: np.ndarray, k: int, batch: int = 16) -> np.ndarray:
    """EXP-002c reference masks (0.5 + largest component) of fold k, exactly as in the pipeline."""
    fm = FoldModels(k, Path("runs"), ["det"], [], views=())
    refs = [fm.infer(images[i : i + batch])[1]["ref"] for i in range(0, len(images), batch)]
    del fm
    return np.concatenate(refs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["clean", "external", "shift"], required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    seed_everything(0, deterministic=True)
    create_run_dir(Path("runs"), "exp011_medsam", {"mode": args.mode}, exist_ok=True)
    sam = MedSAM()

    if args.mode == "clean":
        A = FigshareArrays.load("data/processed/figshare_256")
        for k in args.folds:
            out = OUT / f"clean_fold{k}.csv"
            if out.exists():
                continue
            t0 = time.time()
            split = load_split(Path("data/splits") / f"figshare_fold{k}.json", A.meta)
            pm, _ = load_predicted_masks(Path("runs") / "exp002c_unet_r34" / f"fold{k}", len(A.meta))
            recs = []
            for part, rows in (("val", split.val), ("test", split.test)):
                for r in rows:
                    s = sam.analyse(A.images[r], [pm[r]], [np.random.default_rng([0, int(r), k])], gt=A.masks[r])[0]
                    recs.append({"fold": k, "part": part, "row": int(r), "unet_dice_gt": dice(pm[r], A.masks[r]), **s})
            pd.DataFrame(recs).to_csv(out, index=False)
            print(f"clean fold {k}: {len(recs)} slices in {time.time() - t0:.0f}s", flush=True)
        return

    if args.mode == "external":
        outs = [OUT / f"external_fold{k}.csv" for k in args.folds]
        if all(o.exists() for o in outs):
            return
        t0 = time.time()
        meta = pd.read_csv("data/processed/bdneuro_256/metadata.csv")
        keep = meta.index[meta.audit == "clean"].to_numpy()
        imgs = np.load("data/processed/bdneuro_256/images.npy")[keep]
        refs = {k: unet_refs(imgs, k) for k in args.folds}
        recs = {k: [] for k in args.folds}
        for i, er in enumerate(keep):
            res = sam.analyse(imgs[i], [refs[k][i] for k in args.folds],
                              [np.random.default_rng([1, int(er), k]) for k in args.folds])
            for k, s in zip(args.folds, res):
                recs[k].append({"fold": k, "ext_row": int(er), **s})
        for k, o in zip(args.folds, outs):
            pd.DataFrame(recs[k]).to_csv(o, index=False)
        print(f"external: {len(keep)} images x {len(args.folds)} folds in {time.time() - t0:.0f}s", flush=True)
        return

    A = FigshareArrays.load("data/processed/figshare_256")
    for k in args.folds:
        rows = np.array(load_split(Path("data/splits") / f"figshare_fold{k}.json", A.meta).test)
        for ki, kind in enumerate(KINDS):
            for sev in (1, 2, 3):
                tag = f"{kind}_s{sev}"
                out = OUT / f"shift_{tag}_fold{k}.csv"
                if out.exists():
                    continue
                t0 = time.time()
                imgs = np.stack([corrupt(A.images[r], kind, sev, np.random.default_rng([0, int(r), ki, sev])) for r in rows])
                refs = unet_refs(imgs, k)
                stored = pd.read_csv(Path("runs") / "exp008_shift" / f"{tag}_fold{k}.csv").set_index("row")
                d_now = np.array([dice(refs[i], A.masks[r]) for i, r in enumerate(rows)])
                gap = float(np.nanmax(np.abs(d_now - stored.loc[rows, "dice_lcc"].to_numpy())))
                if gap > 1e-6:
                    raise SystemExit(f"{tag} fold {k}: U-Net Dice differs from EXP-008 by {gap} (pre-specified check)")
                recs = []
                for i, r in enumerate(rows):
                    s = sam.analyse(imgs[i], [refs[i]], [np.random.default_rng([2, int(r), k, ki, sev])], gt=A.masks[r])[0]
                    recs.append({"fold": k, "row": int(r), "unet_dice_gt": float(d_now[i]), **s})
                pd.DataFrame(recs).to_csv(out, index=False)
                print(f"shift {tag} fold {k}: {len(rows)} slices in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
