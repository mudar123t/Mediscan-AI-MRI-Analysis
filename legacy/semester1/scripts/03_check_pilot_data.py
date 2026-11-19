from pathlib import Path
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")
PILOT_ROOT = PROJECT_ROOT / "data" / "pilot" / "processed"

def show_one(split: str):
    img_dir = PILOT_ROOT / "images" / split
    mask_dir = PILOT_ROOT / "masks" / split

    files = list(img_dir.glob("*.png"))
    assert len(files) > 0, f"No images found in {img_dir}"

    f = random.choice(files)
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_dir / f.name), cv2.IMREAD_GRAYSCALE)

    print(f"\n[{split}] sample: {f.name}")
    print("img:", img.shape, img.dtype, "min/max:", img.min(), img.max())
    print("mask:", mask.shape, mask.dtype, "unique:", np.unique(mask))

    img01 = img.astype(np.float32) / 255.0
    mask01 = (mask > 0).astype(np.float32)

    overlay = np.stack([img01, img01, img01], axis=-1)
    overlay[..., 0] = np.maximum(overlay[..., 0], mask01)

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(img01, cmap="gray"); ax[0].set_title("Pilot MRI"); ax[0].axis("off")
    ax[1].imshow(mask01, cmap="gray"); ax[1].set_title("Pilot Mask (binary)"); ax[1].axis("off")
    ax[2].imshow(overlay); ax[2].set_title("Overlay"); ax[2].axis("off")
    plt.tight_layout()
    plt.show()

def main():
    for split in ["train", "val", "test"]:
        show_one(split)

if __name__ == "__main__":
    main()
