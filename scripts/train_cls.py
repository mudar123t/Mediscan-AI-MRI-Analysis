"""M0.4 / EXP-003: train a tumor-type classifier on one input view, per patient-disjoint fold.

Usage (from the repository root):
    .venv/Scripts/python scripts/train_cls.py --config configs/exp003_r18_full.yaml [--folds 0] [--max-epochs N]

Training and model selection use views built from **ground-truth** masks. After
training, the selected model is applied to val and test views built from
(a) GT masks and (b) masks predicted by the EXP-002c segmenter of the same fold
(threshold 0.5, largest component). The segmenter of fold k never saw test fold k.

Per fold, writes ``runs/<run_name>/fold<k>/``:
    config.yaml, environment.json, history.csv, best.pt,
    logits_{val,test}_{gt,pred}.npz (rows, labels, logits; ``pred`` only for mask views),
    summary.json (quick test metrics; the reported numbers come from scripts/eval_cls.py)
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

from mediscan.data.augment import train_transform
from mediscan.data.datasets import FigshareArrays, FigshareClsDataset, load_predicted_masks
from mediscan.data.figshare import CLASS_NAMES
from mediscan.data.splits import load_split
from mediscan.metrics import cls as clsm
from mediscan.models.cls import build_cls_model
from mediscan.utils.optim import lr_lambda, make_optimizer
from mediscan.utils.repro import create_run_dir, load_config, seed_everything
from mediscan.views import build_view, uses_mask


@torch.no_grad()
def predict_logits(model: torch.nn.Module, ds: FigshareClsDataset, device: str, amp: bool, bs: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for batch in DataLoader(ds, batch_size=bs, shuffle=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            logits = model(batch["image"].to(device))
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def labels_of(arrays: FigshareArrays, rows: list[int]) -> np.ndarray:
    return arrays.meta.loc[rows, "class_index"].to_numpy(dtype=np.int64)


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
    vcfg = cfg["view"]
    view_fn = build_view(vcfg["name"], vcfg.get("margin"))

    split = load_split(Path(cfg["paths"]["splits"]) / f"figshare_fold{fold}.json", arrays.meta)
    train_ds = FigshareClsDataset(arrays, split.train, view_fn, "gt", transform=train_transform(tcfg["augment"]))
    val_ds = FigshareClsDataset(arrays, split.val, view_fn, "gt")
    y_val = labels_of(arrays, split.val)
    gen = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, generator=gen)

    model = build_cls_model(cfg["model"]["name"], num_classes=len(CLASS_NAMES), **cfg["model"].get("kwargs", {})).to(device)
    opt = make_optimizer(model, tcfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tcfg, len(train_dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss()
    patience = tcfg.get("early_stopping_patience")
    select = tcfg.get("select_metric", "balanced_accuracy")

    best_key, best_epoch, best_state, history = (-1.0, 0.0), -1, None, []
    t_start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        tr_loss, n = 0.0, 0
        for batch in train_dl:
            x, y = batch["image"].to(device), batch["label"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_loss += float(loss.detach()) * x.size(0)
            n += x.size(0)
        t_train = time.time() - t0

        logits = predict_logits(model, val_ds, device, amp)
        c = clsm.confusion(y_val, logits.argmax(1), len(CLASS_NAMES))
        val_score = float(clsm.COUNT_METRICS[select](c))
        val_nll = clsm.nll(clsm.softmax(logits), y_val)
        history.append({"epoch": epoch, "train_loss": tr_loss / n, f"val_{select}": val_score, "val_nll": val_nll,
                        "lr": opt.param_groups[0]["lr"], "train_s": round(t_train, 1), "epoch_s": round(time.time() - t0, 1)})
        print(f"fold {fold} ep {epoch:3d} loss {tr_loss / n:.4f} val_{select} {val_score:.4f} "
              f"val_nll {val_nll:.4f} ({time.time() - t0:.1f}s)", flush=True)
        key = (val_score, -val_nll)  # ties on the score are broken by lower val NLL
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif patience is not None and epoch - best_epoch >= int(patience):
            print(f"fold {fold}: early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break
    train_time = time.time() - t_start
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best.pt")

    summary: dict[str, Any] = {"fold": fold, "best_epoch": best_epoch, f"best_val_{select}": best_key[0],
                               "epochs_run": len(history), "train_time_s": round(train_time, 1),
                               "peak_vram_mib": int(torch.cuda.max_memory_allocated() // 2**20) if device == "cuda" else None}
    sources: dict[str, Any] = {"gt": "gt"}
    if uses_mask(vcfg["name"]):
        pm = cfg["pred_masks"]
        masks, filled = load_predicted_masks(Path(cfg["paths"]["runs"]) / pm["seg_run"] / f"fold{fold}", len(arrays.meta),
                                             float(pm["threshold"]), bool(pm["keep_largest"]))
        assert set(split.val) | set(split.test) <= set(filled), "segmenter predictions missing for val/test rows"
        sources["pred"] = masks
        summary["pred_mask_empty_test"] = int(sum(masks[r].sum() == 0 for r in split.test))
    for src_name, src in sources.items():
        for part, rows in (("val", split.val), ("test", split.test)):
            logits = predict_logits(model, FigshareClsDataset(arrays, rows, view_fn, src), device, amp)
            y = labels_of(arrays, rows)
            np.savez_compressed(run_dir / f"logits_{part}_{src_name}.npz", rows=np.array(rows), labels=y, logits=logits)
            if part == "test":
                m = clsm.all_metrics(logits, y, list(CLASS_NAMES))
                summary[f"test_{src_name}"] = {k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "auroc_ovr", "ece")}
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
