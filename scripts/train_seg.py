"""M0.3 / EXP-002: train and evaluate a segmentation model on the patient-disjoint folds.

Usage (from the repository root):
    .venv/Scripts/python scripts/train_seg.py --config configs/exp002a_unet_s1_old.yaml [--folds 0 1] [--max-epochs N]

Per fold, writes ``runs/<run_name>/fold<k>/``:
    config.yaml, environment.json, history.csv, best.pt,
    pred_{val,test}.npz (rows + sigmoid probabilities as uint8 0..255),
    metrics_{val,test}.csv (per-image Dice, IoU, HD95 px, NSD@2px; raw and largest-component),
    summary.json
Folds whose ``summary.json`` exists are skipped (resume after interruption).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mediscan.data.augment import train_transform
from mediscan.data.datasets import FigshareArrays, FigshareSegDataset
from mediscan.data.splits import load_split
from mediscan.metrics import seg as segm
from mediscan.models.seg import build_seg_model
from mediscan.segmentation.losses import BCEDiceLoss
from mediscan.utils.optim import lr_lambda, make_optimizer
from mediscan.utils.repro import create_run_dir, load_config, seed_everything


def _batch_dice(logits: torch.Tensor, target: torch.Tensor, thr: float) -> torch.Tensor:
    """Per-image hard Dice (B,) on the GPU, used for model selection only."""
    pred = (torch.sigmoid(logits.float()) > thr).float()
    inter = (pred * target).sum((1, 2, 3))
    denom = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return torch.where(denom > 0, 2 * inter / denom.clamp(min=1e-6), torch.ones_like(denom))


@torch.no_grad()
def predict(model: torch.nn.Module, ds: FigshareSegDataset, device: str, amp: bool, bs: int = 16) -> np.ndarray:
    """Sigmoid probabilities (N, H, W) as float32, deterministic (dropout off)."""
    model.eval()
    out = []
    for batch in DataLoader(ds, batch_size=bs, shuffle=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            logits = model(batch["image"].to(device))
        out.append(torch.sigmoid(logits.float())[:, 0].cpu().numpy())
    return np.concatenate(out)


def evaluate(probs: np.ndarray, arrays: FigshareArrays, rows: list[int], thr: float) -> pd.DataFrame:
    recs = []
    for p, row in zip(probs, rows):
        gt = arrays.masks[row]
        pred = (p > thr).astype(np.uint8)
        m = segm.all_metrics(pred, gt)
        lcc = segm.largest_component(pred)
        m["dice_lcc"] = segm.dice(lcc, gt)
        m["hd95_lcc"] = segm.hd95(lcc, gt)
        meta = arrays.meta.loc[row]
        recs.append({"row": row, "file": int(meta["file"]), "pid": meta["pid"], "class_name": meta["class_name"], **m})
    return pd.DataFrame(recs)


def train_fold(cfg: dict[str, Any], arrays: FigshareArrays, fold: int, run_root: Path, max_epochs: int | None) -> None:
    tcfg = dict(cfg["train"])
    if max_epochs is not None:
        tcfg["epochs"] = min(tcfg["epochs"], max_epochs)
    fold_cfg = {**cfg, "train": tcfg, "fold": fold}
    run_dir = create_run_dir(run_root, f"fold{fold}", fold_cfg, exist_ok=True)
    seed = int(cfg["seed"])
    seed_everything(seed, deterministic=bool(cfg.get("deterministic", True)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(tcfg.get("amp", False)) and device == "cuda"
    thr = float(cfg["eval"]["threshold"])

    split = load_split(Path(cfg["paths"]["splits"]) / f"figshare_fold{fold}.json", arrays.meta)
    train_ds = FigshareSegDataset(arrays, split.train, transform=train_transform(tcfg["augment"]))
    val_ds = FigshareSegDataset(arrays, split.val)
    test_ds = FigshareSegDataset(arrays, split.test)
    gen = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, generator=gen)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = build_seg_model(cfg["model"]["name"], **cfg["model"].get("kwargs", {})).to(device)
    opt = make_optimizer(model, tcfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tcfg, len(train_dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = BCEDiceLoss(bce_weight=float(tcfg.get("bce_weight", 0.5)))
    patience = tcfg.get("early_stopping_patience")

    best_dice, best_epoch, best_state, history = -1.0, -1, None, []
    t_start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        tr_loss, n = 0.0, 0
        for batch in train_dl:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_loss += float(loss.detach()) * x.size(0)
            n += x.size(0)
        t_train = time.time() - t0

        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_dl:
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(batch["image"].to(device))
                dices.append(_batch_dice(logits, batch["mask"].to(device), thr).cpu())
        val_dice = float(torch.cat(dices).mean())
        history.append(
            {"epoch": epoch, "train_loss": tr_loss / n, "val_dice": val_dice, "lr": opt.param_groups[0]["lr"],
             "train_s": round(t_train, 1), "epoch_s": round(time.time() - t0, 1)}
        )
        print(f"fold {fold} ep {epoch:3d} loss {tr_loss / n:.4f} val_dice {val_dice:.4f} ({time.time() - t0:.1f}s)", flush=True)
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif patience is not None and epoch - best_epoch >= int(patience):
            print(f"fold {fold}: early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break
    train_time = time.time() - t_start
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best.pt")

    summary: dict[str, Any] = {"fold": fold, "best_epoch": best_epoch, "best_val_dice_selection": best_dice,
                               "epochs_run": len(history), "train_time_s": round(train_time, 1),
                               "peak_vram_mib": int(torch.cuda.max_memory_allocated() // 2**20) if device == "cuda" else None}
    for part, ds, rows in (("val", val_ds, split.val), ("test", test_ds, split.test)):
        probs = predict(model, ds, device, amp)
        np.savez_compressed(run_dir / f"pred_{part}.npz", rows=np.array(rows),
                            probs=np.round(probs * 255).astype(np.uint8))
        df = evaluate(probs, arrays, rows, thr)
        df.to_csv(run_dir / f"metrics_{part}.csv", index=False)
        summary[part] = {k: float(df[k].mean()) for k in ("dice", "iou", "hd95", "nsd", "dice_lcc", "hd95_lcc")}
        summary[part]["frac_empty_pred"] = float((df["pred_area"] == 0).mean())
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="cap epochs (timing probes only)")
    parser.add_argument("--run-name", default=None, help="override run folder name")
    args = parser.parse_args()

    base = load_config("configs/base.yaml")
    cfg = load_config(args.config)
    cfg["paths"] = {**base["paths"], "splits": "data/splits", "processed": "data/processed/figshare_256", **cfg.get("paths", {})}
    run_root = Path(cfg["paths"]["runs"]) / (args.run_name or cfg["run_name"])
    arrays = FigshareArrays.load(cfg["paths"]["processed"])
    for fold in args.folds if args.folds is not None else cfg["folds"]:
        if (run_root / f"fold{fold}" / "summary.json").exists():
            print(f"fold {fold}: summary.json exists, skipping")
            continue
        train_fold(cfg, arrays, fold, run_root, args.max_epochs)


if __name__ == "__main__":
    main()
