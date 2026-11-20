"""M1.1 / EXP-004 (R3): classifier output vs controlled mask quality, for every mask-based view.

The EXP-003 ResNet-18 view models (trained on GT-mask views) are applied, unchanged,
to test-fold views built from GT masks degraded to target Dice levels
(``mediscan.degrade``). Inference only; no training.

Usage:
    .venv/Scripts/python scripts/run_exp004.py --config configs/exp004_mask_quality.yaml

Writes ``runs/<run_name>/fold<k>.npz`` with
    rows, labels (N,), kinds, targets (C,), achieved_dice (C, N),
    view_runs (V,), logits (V, C, N, 3)
Cell 0 is the undegraded GT mask (kind "none", target 1.0). Existing fold files are skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from mediscan.data.datasets import FigshareArrays, FigshareClsDataset
from mediscan.data.splits import load_split
from mediscan.degrade import degrade
from mediscan.models.cls import build_cls_model
from mediscan.utils.repro import create_run_dir, load_config, seed_everything
from mediscan.views import build_view

import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_cls import predict_logits  # noqa: E402  (same inference code as EXP-003)


def load_view_model(fold_dir: Path, device: str) -> tuple[torch.nn.Module, object, bool]:
    cfg = load_config(fold_dir / "config.yaml")
    model = build_cls_model(cfg["model"]["name"], **{**cfg["model"].get("kwargs", {}), "pretrained": False})
    model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device))
    model.to(device).eval()
    view = build_view(cfg["view"]["name"], cfg["view"].get("margin"))
    return model, view, bool(cfg["train"].get("amp", False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    args = p.parse_args()

    base = load_config("configs/base.yaml")
    cfg = load_config(args.config)
    cfg["paths"] = {**base["paths"], "splits": "data/splits", "processed": "data/processed/figshare_256", **cfg.get("paths", {})}
    runs = Path(cfg["paths"]["runs"])
    out_root = runs / cfg["run_name"]
    create_run_dir(runs, cfg["run_name"], cfg, exist_ok=True)
    seed = int(cfg["seed"])
    seed_everything(seed, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = FigshareArrays.load(cfg["paths"]["processed"])
    kinds = ["none"] + [k for k in cfg["kinds"] for _ in cfg["targets"]]
    targets = [1.0] + [float(t) for _ in cfg["kinds"] for t in cfg["targets"]]

    for fold in args.folds if args.folds is not None else cfg["folds"]:
        out = out_root / f"fold{fold}.npz"
        if out.exists():
            print(f"fold {fold}: exists, skipping")
            continue
        t0 = time.time()
        split = load_split(Path(cfg["paths"]["splits"]) / f"figshare_fold{fold}.json", arrays.meta)
        rows = list(split.test)
        labels = arrays.meta.loc[rows, "class_index"].to_numpy(dtype=np.int64)
        models = [load_view_model(runs / v / f"fold{fold}", device) for v in cfg["view_runs"]]
        achieved = np.zeros((len(kinds), len(rows)), dtype=np.float32)
        logits = np.zeros((len(models), len(kinds), len(rows), 3), dtype=np.float32)
        full_masks = np.zeros_like(arrays.masks)
        for c, (kind, target) in enumerate(zip(kinds, targets)):
            for i, r in enumerate(rows):
                # one generator per (seed, row, cell): results do not depend on processing order
                rng = np.random.default_rng([seed, r, c])
                full_masks[r], achieved[c, i] = degrade(arrays.masks[r], kind, target, rng, arrays.images[r])
            for v, (model, view, amp) in enumerate(models):
                ds = FigshareClsDataset(arrays, rows, view, full_masks)
                logits[v, c] = predict_logits(model, ds, device, amp)
            print(f"fold {fold} cell {c:2d} {kind:6s} target {target:.2f} achieved {achieved[c].mean():.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(out, rows=np.array(rows), labels=labels, kinds=np.array(kinds), targets=np.array(targets),
                            achieved_dice=achieved, view_runs=np.array(cfg["view_runs"]), logits=logits)
        print(f"fold {fold}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
