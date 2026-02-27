import os
from pathlib import Path
import random

import numpy as np
import cv2

from mediscan.preprocessing.basic import (
    normalize_01_uint8,
    binarize_mask,
    gaussian_denoise,
    to_uint8
)

# ----------------------------
# CONFIG (edit path if needed)
# ----------------------------
PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")  # <-- change if needed
RAW_ROOT = PROJECT_ROOT / "dataset" / "output"
CLASSES = ["glioma", "meningioma", "pituitary"]

IMG_DIRNAME = "images"
MASK_DIRNAME = "masks"

# Pilot size total (your choice A)
PILOT_TOTAL = 500

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Optional denoise
USE_DENOISE = True
DENOISE_KSIZE = 3

# Output
OUT_ROOT = PROJECT_ROOT / "data" / "pilot"
OUT_IMG = OUT_ROOT / "processed" / "images"
OUT_MASK = OUT_ROOT / "processed" / "masks"
OUT_SPLITS = OUT_ROOT / "splits"

IMG_EXTS = [".png", ".jpg", ".jpeg"]
MASK_EXTS = [".png", ".jpg", ".jpeg"]

SEED = 42


def find_by_stem(folder: Path, stem: str, exts) -> Path | None:
    for e in exts:
        p = folder / f"{stem}{e}"
        if p.exists():
            return p
    return None


def collect_all_pairs():
    """
    Returns list of dicts:
      {"cls": class_name, "stem": stem, "img": img_path, "mask": mask_path}
    """
    items = []
    for cls in CLASSES:
        cdir = RAW_ROOT / cls
        img_dir = cdir / IMG_DIRNAME
        mask_dir = cdir / MASK_DIRNAME

        img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        img_stems = [p.stem for p in img_files]

        for stem in img_stems:
            ip = find_by_stem(img_dir, stem, IMG_EXTS)
            mp = find_by_stem(mask_dir, stem, MASK_EXTS)
            if ip is not None and mp is not None:
                items.append({"cls": cls, "stem": stem, "img": ip, "mask": mp})
    return items


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUT_IMG / split).mkdir(parents=True, exist_ok=True)
        (OUT_MASK / split).mkdir(parents=True, exist_ok=True)
    OUT_SPLITS.mkdir(parents=True, exist_ok=True)


def save_split_list(name: str, rows: list[str]):
    (OUT_SPLITS / f"{name}.txt").write_text("\n".join(rows), encoding="utf-8")


def process_and_save(item, split_name: str):
    # Read grayscale
    img = cv2.imread(str(item["img"]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise FileNotFoundError(f"Read failed for {item['img']} or {item['mask']}")
    if img.shape != mask.shape:
        raise ValueError(f"Shape mismatch for {item['img'].name}: img{img.shape} mask{mask.shape}")

    # Preprocess
    img01 = normalize_01_uint8(img)
    if USE_DENOISE:
        img01 = gaussian_denoise(img01, ksize=DENOISE_KSIZE)

    mask_bin = binarize_mask(mask)  # {0,255}

    # Save as uint8 PNG
    out_name = f"{item['cls']}_{item['stem']}.png"
    out_img_path = OUT_IMG / split_name / out_name
    out_mask_path = OUT_MASK / split_name / out_name

    cv2.imwrite(str(out_img_path), to_uint8(img01))
    cv2.imwrite(str(out_mask_path), mask_bin)

    # Return relative id for split files
    return out_name


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT not found: {RAW_ROOT}")

    ensure_dirs()

    all_items = collect_all_pairs()
    if len(all_items) < PILOT_TOTAL:
        raise ValueError(f"Not enough paired samples: found {len(all_items)}, need {PILOT_TOTAL}")

    # Shuffle and pick pilot subset
    random.shuffle(all_items)
    pilot_items = all_items[:PILOT_TOTAL]

    # Split sizes
    n_train = int(PILOT_TOTAL * TRAIN_RATIO)
    n_val = int(PILOT_TOTAL * VAL_RATIO)
    n_test = PILOT_TOTAL - n_train - n_val

    train_items = pilot_items[:n_train]
    val_items = pilot_items[n_train:n_train + n_val]
    test_items = pilot_items[n_train + n_val:]

    print("=== Step 2: Prepare Pilot Dataset ===")
    print(f"Total available pairs: {len(all_items)}")
    print(f"Pilot total: {PILOT_TOTAL} -> train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")
    print(f"Output root: {OUT_ROOT}")

    train_ids, val_ids, test_ids = [], [], []

    for it in train_items:
        train_ids.append(process_and_save(it, "train"))
    for it in val_items:
        val_ids.append(process_and_save(it, "val"))
    for it in test_items:
        test_ids.append(process_and_save(it, "test"))

    save_split_list("train", train_ids)
    save_split_list("val", val_ids)
    save_split_list("test", test_ids)

    print(" Done.")
    print(f"Saved train/val/test lists in: {OUT_SPLITS}")
    print(f"Saved images in: {OUT_IMG}")
    print(f"Saved masks  in: {OUT_MASK}")

    # Quick sanity prints
    sample = random.choice(train_ids)
    print(f"Sanity sample id: {sample}")
    print("Next: run scripts/03_check_pilot_data.py")


if __name__ == "__main__":
    main()
