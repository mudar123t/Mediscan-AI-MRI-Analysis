"""EXP-000: inference-only re-evaluation of the semester-1 checkpoints.

No training. Uses the checkpoints in outputs/pilot/checkpoints and the pilot
directories exactly as semester 1 left them. Reports metrics on three subsets
of each split directory:
  dir110  - every file in the directory (what semester-1 scripts evaluated)
  clean50 - only files listed in data/pilot/splits/<split>.txt
  leaked60 - files in the directory that are NOT in the list (all are also in train/)

Run from repo root:  PYTHONPATH=src python project-log/experiments/EXP-000-semester1-checkpoint-eval/eval_semester1.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from mediscan.classification.cnn import SimpleCNN  # noqa: E402
from mediscan.segmentation.unet import UNet  # noqa: E402

PILOT = ROOT / "data" / "pilot"
CKPT = ROOT / "outputs" / "pilot" / "checkpoints"
OUT = Path(__file__).resolve().parent / "results.json"
CLASSES = ["glioma", "meningioma", "pituitary"]
torch.set_grad_enabled(False)


def listed(split: str) -> set[str]:
    return {l.strip() for l in (PILOT / "splits" / f"{split}.txt").read_text().splitlines() if l.strip()}


def subsets(split: str) -> dict[str, list[str]]:
    files = sorted(p.name for p in (PILOT / "processed" / "images" / split).glob("*.png"))
    lst = listed(split)
    return {"dir110": files, "clean50": [f for f in files if f in lst], "leaked60": [f for f in files if f not in lst]}


def gray01(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def seg_eval(model: UNet, split: str, names: list[str]) -> dict:
    dices, ious = [], []
    for n in names:
        x = torch.from_numpy(gray01(PILOT / "processed" / "images" / split / n))[None, None]
        gt = gray01(PILOT / "processed" / "masks" / split / n) > 0
        pred = (torch.sigmoid(model(x))[0, 0].numpy() > 0.5)
        inter = np.logical_and(pred, gt).sum()
        s = pred.sum() + gt.sum()
        dices.append((2 * inter + 1e-6) / (s + 1e-6))
        ious.append((inter + 1e-6) / (s - inter + 1e-6))
    d, i = np.array(dices), np.array(ious)
    return {"n": len(names), "dice_mean": float(d.mean()), "dice_std": float(d.std()),
            "dice_median": float(np.median(d)), "iou_mean": float(i.mean()),
            "frac_dice_below_0.5": float((d < 0.5).mean())}


def cls_eval(model: SimpleCNN, img_dir: Path, names: list[str]) -> dict:
    y_true, y_pred = [], []
    for n in names:
        x = torch.from_numpy(gray01(img_dir / n))[None, None]
        y_pred.append(int(model(x).argmax(1)))
        y_true.append(CLASSES.index(n.split("_")[0]))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {"n": len(names), "accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
            "recall_per_class": dict(zip(CLASSES, map(float, recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)))),
            "support": dict(zip(CLASSES, map(int, np.bincount(y_true, minlength=3)))),
            "confusion_matrix_rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()}


def main() -> None:
    t0 = time.time()
    unet = UNet(); unet.load_state_dict(torch.load(CKPT / "unet_best.pt", map_location="cpu", weights_only=True)); unet.eval()
    res = {"env": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                   "platform": platform.platform()}, "segmentation": {}, "classification": {}}
    for split in ["val", "test"]:
        for k, names in subsets(split).items():
            res["segmentation"][f"{split}/{k}"] = seg_eval(unet, split, names)
    inputs = {"full": PILOT / "processed" / "images", "roi_gt": PILOT / "roi_gt" / "images", "roi_pred": PILOT / "roi_pred" / "images"}
    for name, base in inputs.items():
        m = SimpleCNN(); m.load_state_dict(torch.load(CKPT / f"cnn_{name}_best.pt", map_location="cpu", weights_only=True)); m.eval()
        for split in ["val", "test"]:
            for k, names in subsets(split).items():
                res["classification"][f"{name}/{split}/{k}"] = cls_eval(m, base / split, names)
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
