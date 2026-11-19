"""M0.2 / EXP-001: convert Figshare .mat files to arrays, metadata and patient-disjoint splits.

Usage (from the repository root):
    .venv/Scripts/python scripts/prepare_figshare.py --config configs/data_figshare.yaml

Writes (git-ignored):  <processed>/images.npy, masks.npy, metadata.csv, report.json, config.yaml
Writes (tracked):      <splits>/figshare_fold{0..4}.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from mediscan.data import figshare as fs
from mediscan.data.splits import N_FOLDS, make_split, save_split
from mediscan.utils.repro import environment_info, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_figshare.yaml")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing processed folder")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = Path(cfg["paths"]["raw_data"])
    out = Path(cfg["paths"]["processed"])
    split_dir = Path(cfg["paths"]["splits"])
    size = int(cfg["data"]["image_size"])
    lo_pct, hi_pct = cfg["data"]["intensity_percentiles"]
    min_iou = float(cfg["checks"]["min_border_iou"])

    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite to rebuild it")
    out.mkdir(parents=True, exist_ok=True)

    files = sorted((raw / "data").glob("*.mat"), key=lambda p: int(p.stem))
    numbers = [int(p.stem) for p in files]
    if numbers != list(range(1, len(files) + 1)):
        raise SystemExit("expected files 1.mat .. N.mat without gaps")
    folds = fs.read_folds(raw / "cvind.mat")
    if len(folds) != len(files):
        raise SystemExit(f"cvind has {len(folds)} entries but there are {len(files)} files")

    t0 = time.time()
    images = np.zeros((len(files), size, size), dtype=np.uint8)
    masks = np.zeros((len(files), size, size), dtype=np.uint8)
    rows, low_iou = [], []
    native_sizes: Counter[str] = Counter()
    hashes: dict[str, list[int]] = {}

    for row, path in enumerate(tqdm(files, desc="convert")):
        case = fs.read_case(path)
        h, w = case.image.shape
        native_sizes[f"{h}x{w}"] += 1
        if case.mask.shape != case.image.shape:
            raise SystemExit(f"{path}: mask/image shape mismatch")
        if not set(np.unique(case.mask)) <= {0, 1}:
            raise SystemExit(f"{path}: mask is not binary")
        border_iou = fs.iou(fs.border_to_mask(case.border_xy, case.mask.shape), case.mask)
        if border_iou < min_iou:
            low_iou.append((case.index, round(border_iou, 3)))
        digest = hashlib.sha1(case.image.tobytes()).hexdigest()
        hashes.setdefault(digest, []).append(case.index)

        img01 = fs.normalize_intensity(case.image, lo_pct, hi_pct)
        img_r, msk_r = fs.resize_pair(img01, case.mask, size)
        images[row] = fs.to_uint8(img_r)
        masks[row] = msk_r

        ys, xs = np.nonzero(msk_r)
        rows.append(
            {
                "file": case.index,
                "pid": case.pid,
                "label": case.label,
                "class_index": case.class_index,
                "class_name": fs.CLASS_NAMES[case.class_index],
                "fold": int(folds[row]) - 1,  # 0-based
                "native_h": h,
                "native_w": w,
                "mask_area_native": int(case.mask.sum()),
                "mask_area": int(msk_r.sum()),
                "bbox_y0": int(ys.min()) if ys.size else -1,
                "bbox_x0": int(xs.min()) if xs.size else -1,
                "bbox_y1": int(ys.max()) if ys.size else -1,
                "bbox_x1": int(xs.max()) if xs.size else -1,
                "border_iou": round(border_iou, 4),
                "image_sha1": digest,
            }
        )

    meta = pd.DataFrame(rows)
    np.save(out / "images.npy", images)
    np.save(out / "masks.npy", masks)
    meta.to_csv(out / "metadata.csv", index=False)
    with open(out / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    split_sizes = {}
    for k in range(N_FOLDS):
        split = make_split(meta, k)  # raises LeakageError on patient overlap
        save_split(split, split_dir / f"figshare_fold{k}.json")
        split_sizes[k] = {
            part: {
                "slices": len(getattr(split, part)),
                "patients": int(meta.loc[getattr(split, part), "pid"].nunique()),
            }
            for part in ("train", "val", "test")
        }

    pid_folds = meta.groupby("pid")["fold"].nunique()
    duplicates = {h: idx for h, idx in hashes.items() if len(idx) > 1}
    report = {
        "n_slices": len(meta),
        "n_patients": int(meta["pid"].nunique()),
        "slices_per_class": meta["class_name"].value_counts().to_dict(),
        "patients_per_class": meta.groupby("class_name")["pid"].nunique().to_dict(),
        "patients_with_multiple_classes": int((meta.groupby("pid")["label"].nunique() > 1).sum()),
        "slices_per_fold": meta["fold"].value_counts().sort_index().to_dict(),
        "patients_per_fold": meta.groupby("fold")["pid"].nunique().to_dict(),
        "patients_in_more_than_one_fold": int((pid_folds > 1).sum()),
        "class_slices_per_fold": pd.crosstab(meta["fold"], meta["class_name"]).to_dict(orient="index"),
        "native_sizes": dict(native_sizes),
        "border_iou": {
            "min": float(meta["border_iou"].min()),
            "p01": float(meta["border_iou"].quantile(0.01)),
            "median": float(meta["border_iou"].median()),
            f"n_below_{min_iou}": len(low_iou),
            "cases_below": low_iou[:50],
        },
        "empty_masks_native": int((meta["mask_area_native"] == 0).sum()),
        "empty_masks_after_resize": int((meta["mask_area"] == 0).sum()),
        "mask_area_fraction_256": meta["mask_area"].div(size * size).describe().round(5).to_dict(),
        "exact_duplicate_image_groups": list(duplicates.values()),
        "split_sizes": split_sizes,
        "runtime_s": round(time.time() - t0, 1),
        "environment": environment_info(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=1, default=str))


if __name__ == "__main__":
    main()
