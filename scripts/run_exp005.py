"""M1.2 / EXP-005: image-level segmentation-uncertainty scores for val and test slices of every fold.

Methods:
* ``det``: the deterministic EXP-002c prediction (1 sample; no-sampling baseline);
* ``mc``: MC dropout with the EXP-002c model, T samples (decoder Dropout2d active);
* ``tta``: the 8 dihedral test-time augmentations of the EXP-002c model;
* ``ens``: seed ensemble, the stored probability maps of the seed runs (no GPU).
The reference mask is the EXP-002c prediction at 0.5 + largest component (what the
classifiers use). ``ens`` is computed only when every seed run has the fold.

Usage:
    .venv/Scripts/python scripts/run_exp005.py --config configs/exp005_uncertainty.yaml [--methods det mc tta ens]

Writes ``runs/<run_name>/scores_fold<k>_<method>.csv`` (row, part, score columns).
Existing files are skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mediscan.data.datasets import FigshareArrays, load_predicted_masks
from mediscan.data.splits import load_split
from mediscan.models.seg import build_seg_model, enable_mc_dropout
from mediscan.uncertainty import image_scores, mc_dropout_samples, tta_samples
from mediscan.utils.repro import create_run_dir, load_config, seed_everything


def stored_probs(run_dir: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for part in ("val", "test"):
        z = np.load(run_dir / f"pred_{part}.npz")
        out.update({int(r): p.astype(np.float32) / 255.0 for r, p in zip(z["rows"], z["probs"])})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--methods", nargs="*", default=["det", "mc", "tta", "ens"])
    args = p.parse_args()

    base = load_config("configs/base.yaml")
    cfg = load_config(args.config)
    cfg["paths"] = {**base["paths"], "splits": "data/splits", "processed": "data/processed/figshare_256", **cfg.get("paths", {})}
    runs = Path(cfg["paths"]["runs"])
    out_root = runs / cfg["run_name"]
    create_run_dir(runs, cfg["run_name"], cfg, exist_ok=True)
    seed_everything(int(cfg["seed"]), deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    arrays = FigshareArrays.load(cfg["paths"]["processed"])
    seg_run = runs / cfg["seg_run"]

    for fold in args.folds if args.folds is not None else cfg["folds"]:
        split = load_split(Path(cfg["paths"]["splits"]) / f"figshare_fold{fold}.json", arrays.meta)
        rows = [("val", r) for r in split.val] + [("test", r) for r in split.test]
        ref_masks, _ = load_predicted_masks(seg_run / f"fold{fold}", len(arrays.meta), 0.5, True)
        for method in args.methods:
            out = out_root / f"scores_fold{fold}_{method}.csv"
            if out.exists():
                print(f"fold {fold} {method}: exists, skipping")
                continue
            t0 = time.time()
            recs: list[dict] = []
            if method in ("det", "ens"):
                dirs = [seg_run] if method == "det" else [runs / r for r in cfg["ensemble_runs"]]
                if not all((d / f"fold{fold}" / "summary.json").exists() for d in dirs):
                    print(f"fold {fold} {method}: seed runs not finished, skipping")
                    continue
                maps = [stored_probs(d / f"fold{fold}") for d in dirs]
                for part, r in rows:
                    recs.append({"row": r, "part": part, **image_scores(np.stack([m[r] for m in maps]), ref_masks[r])})
            else:
                model = build_seg_model("unet_r34", pretrained=False, dropout=float(cfg["dropout"])).to(device)
                model.load_state_dict(torch.load(seg_run / f"fold{fold}" / "best.pt", map_location=device))
                if method == "mc":
                    enable_mc_dropout(model)
                else:
                    model.eval()
                bs = int(cfg["batch_size"])
                for i in range(0, len(rows), bs):
                    chunk = rows[i : i + bs]
                    x = torch.from_numpy(np.stack([arrays.images[r] for _, r in chunk]).astype(np.float32) / 255.0)
                    x = x[:, None].to(device)
                    s = mc_dropout_samples(model, x, int(cfg["mc_samples"]), amp) if method == "mc" else tta_samples(model, x, amp)
                    s = s.cpu().numpy()
                    for j, (part, r) in enumerate(chunk):
                        recs.append({"row": r, "part": part, **image_scores(s[:, j], ref_masks[r])})
            pd.DataFrame(recs).to_csv(out, index=False)
            print(f"fold {fold} {method}: {len(recs)} images in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
