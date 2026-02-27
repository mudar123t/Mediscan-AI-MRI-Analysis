import os
from pathlib import Path
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG (edit this path only)
# ----------------------------
PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")  
CLASSES = ["glioma", "meningioma", "pituitary"]

IMG_DIRNAME = "images"
MASK_DIRNAME = "masks"

# How many samples to visualize per class
SAMPLES_PER_CLASS = 5

# If your files are not png, add extensions here:
IMG_EXTS = [".png", ".jpg", ".jpeg"]
MASK_EXTS = [".png", ".jpg", ".jpeg"]


# ----------------------------
# Helpers
# ----------------------------
def imread_grayscale(path: Path) -> np.ndarray:
    """Read image as grayscale uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def normalize_01(img: np.ndarray) -> np.ndarray:
    """Normalize uint8/float image to [0,1] for plotting."""
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def find_by_stem(folder: Path, stem: str, exts) -> Path | None:
    """Find a file in folder that matches stem + one of the extensions."""
    for e in exts:
        p = folder / f"{stem}{e}"
        if p.exists():
            return p
    return None


def overlay_mask_on_image(img01: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """
    Creates a simple overlay visualization:
    - base: grayscale image in RGB
    - mask: shown in red
    """
    rgb = np.stack([img01, img01, img01], axis=-1)
    overlay = rgb.copy()
    # Put mask into red channel
    overlay[..., 0] = np.maximum(overlay[..., 0], mask01 * 1.0)
    return overlay


def check_pairs(class_dir: Path) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    """
    Returns:
      pairs: list of (image_path, mask_path)
      missing_masks: stems where image exists but mask missing
      missing_images: stems where mask exists but image missing
    """
    img_dir = class_dir / IMG_DIRNAME
    mask_dir = class_dir / MASK_DIRNAME

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing images/ or masks/ in {class_dir}")

    # Collect stems
    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    mask_files = [p for p in mask_dir.iterdir() if p.suffix.lower() in MASK_EXTS]

    img_stems = set(p.stem for p in img_files)
    mask_stems = set(p.stem for p in mask_files)

    missing_masks = sorted(list(img_stems - mask_stems))
    missing_images = sorted(list(mask_stems - img_stems))

    common = sorted(list(img_stems & mask_stems))

    pairs = []
    for stem in common:
        ip = find_by_stem(img_dir, stem, IMG_EXTS)
        mp = find_by_stem(mask_dir, stem, MASK_EXTS)
        if ip is not None and mp is not None:
            pairs.append((ip, mp))

    return pairs, missing_masks, missing_images


def show_pair(img_path: Path, mask_path: Path, title: str = "") -> None:
    img = imread_grayscale(img_path)
    mask = imread_grayscale(mask_path)

    # Basic checks
    same_shape = (img.shape == mask.shape)

    # Normalize for plotting
    img01 = normalize_01(img)

    # Convert mask to binary for overlay (handles 0/255 or 0/1 etc.)
    # threshold > 0 -> tumor
    mask_bin = (mask > 0).astype(np.float32)

    overlay = overlay_mask_on_image(img01, mask_bin)

    # Print info
    uniq = np.unique(mask)
    print(f"\n[{title}]")
    print(f"Image: {img_path.name}  shape={img.shape}  dtype={img.dtype}  min={img.min()} max={img.max()}")
    print(f"Mask : {mask_path.name} shape={mask.shape} dtype={mask.dtype} unique_values={uniq[:20]}{'...' if len(uniq)>20 else ''}")
    print(f"Aligned shapes? {same_shape}")

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img01, cmap="gray")
    ax[0].set_title("MRI (normalized)")
    ax[0].axis("off")

    ax[1].imshow(mask_bin, cmap="gray")
    ax[1].set_title("Mask (binary)")
    ax[1].axis("off")

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay (mask in red)")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

from pathlib import Path

# ----------------------------
# CONFIG (edit this path only)
# ----------------------------
PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")
DATA_ROOT = PROJECT_ROOT / "dataset" / "output"

CLASSES = ["glioma", "meningioma", "pituitary"]

def main():
    print("=== Step 1: Dataset Exploration ===")
    print(f"DATA_ROOT: {DATA_ROOT}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"DATA_ROOT not found: {DATA_ROOT}\n"
            f"Edit PROJECT_ROOT in this script to your correct path."
        )

    for cls in CLASSES:
        class_dir = DATA_ROOT / cls
        print(f"\n--- Class: {cls} ---")
        pairs, missing_masks, missing_images = check_pairs(class_dir)

        print(f"Total paired samples: {len(pairs)}")
        if missing_masks:
            print(f"⚠ Missing masks for {len(missing_masks)} images. Example stems: {missing_masks[:10]}")
        if missing_images:
            print(f"⚠ Missing images for {len(missing_images)} masks. Example stems: {missing_images[:10]}")

        if len(pairs) == 0:
            print("No pairs found — check folder names and extensions.")
            continue

        # sample
        k = min(SAMPLES_PER_CLASS, len(pairs))
        sample_pairs = random.sample(pairs, k)

        for i, (ip, mp) in enumerate(sample_pairs, 1):
            show_pair(ip, mp, title=f"{cls} sample {i}/{k}")


if __name__ == "__main__":
    main()
