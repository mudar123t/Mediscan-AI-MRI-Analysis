"""EXP-010 (R4): Grad-CAM attribution of the EXP-003 crop-32 and full-image classifiers, val + test, 5 folds.

Pre-specified in project-log/experiments/EXP-010-xai-diagnostic.md. No training.
Writes runs/exp010_xai/fold{k}.csv (one row per slice x view) and cams_test_fold{k}.npz
(64x64 float16 back-projected test maps, for figures). Stops if the logits do not
reproduce the stored EXP-003 logits.

Usage:
    .venv/Scripts/python scripts/run_exp010.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from mediscan.data.datasets import FigshareArrays, load_predicted_masks
from mediscan.data.splits import load_split
from mediscan.pipeline import _cls
from mediscan.utils.repro import create_run_dir, seed_everything
from mediscan.views import crop_box
from mediscan.xai import backproject, gradcam, region_fractions

VIEWS = {"crop32": ("exp003_r18_crop32", "pred", 32), "full": ("exp003_r18_full", "gt", None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ring", type=int, default=16)
    args = ap.parse_args()
    runs = Path("runs")
    seed_everything(0, deterministic=True)
    create_run_dir(runs, "exp010_xai", vars(args), exist_ok=True)
    out_dir = runs / "exp010_xai"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = FigshareArrays.load("data/processed/figshare_256")
    n = len(arrays.meta)
    for k in args.folds:
        out = out_dir / f"fold{k}.csv"
        if out.exists():
            print(f"fold {k}: exists, skipped")
            continue
        t0 = time.time()
        split = load_split(Path("data/splits") / f"figshare_fold{k}.json", arrays.meta)
        pred_masks, _ = load_predicted_masks(runs / "exp002c_unet_r34" / f"fold{k}", n)
        recs, cams_test = [], {}
        for v, (run, src, margin) in VIEWS.items():
            model, view = _cls(runs / run / f"fold{k}", device)
            model.float()
            layer = model.net.layer4
            for part, rows in (("val", np.array(split.val)), ("test", np.array(split.test))):
                stored = np.load(runs / run / f"fold{k}" / f"logits_{part}_{src}.npz")
                stored_logits = dict(zip(stored["rows"].tolist(), stored["logits"]))
                for i in range(0, len(rows), args.batch):
                    rb = rows[i : i + args.batch]
                    imgs, masks = arrays.images[rb], pred_masks[rb]
                    xv = np.stack([view(im, mk) for im, mk in zip(imgs, masks)]).astype(np.float32) / 255.0
                    cam, logits = gradcam(model, layer, torch.from_numpy(xv)[:, None].to(device))
                    for j, r in enumerate(rb):
                        box = crop_box(masks[j], margin) if margin is not None else None
                        c = backproject(cam[j], box, imgs[j].shape)
                        fg = region_fractions(c, arrays.masks[r], args.ring)
                        fp = region_fractions(c, masks[j], args.ring)
                        ref = stored_logits[int(r)]
                        recs.append({"fold": k, "part": part, "row": int(r), "y": int(arrays.meta.loc[r, "class_index"]),
                                     "view": v, "pred": int(logits[j].argmax()), "stored_pred": int(ref.argmax()),
                                     "max_abs_logit_diff": float(np.abs(logits[j] - ref).max()),
                                     **{f"l_{c_}": float(logits[j, c_]) for c_ in range(3)},
                                     "cam_mass": float(c.sum()), "gt_in": fg[0], "gt_ring": fg[1], "gt_out": fg[2],
                                     "pred_in": fp[0], "pred_ring": fp[1], "pred_out": fp[2],
                                     "pred_area": int(masks[j].sum())})
                        if part == "test":
                            cams_test[f"{v}_{int(r)}"] = cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float16)
            del model
            torch.cuda.empty_cache()
        df = pd.DataFrame(recs)
        diff, agree = df.max_abs_logit_diff.max(), (df.pred == df.stored_pred).mean()
        print(f"fold {k}: {len(df)} rows in {time.time() - t0:.0f}s; max |logit diff| {diff:.4f}; argmax agreement {agree:.4f}",
              flush=True)
        if diff >= 0.05 or agree < 0.999:
            raise SystemExit(f"fold {k}: logits do not reproduce EXP-003 (pre-specified check); stopping")
        df.to_csv(out, index=False)
        np.savez_compressed(out_dir / f"cams_test_fold{k}.npz", **cams_test)


if __name__ == "__main__":
    main()
