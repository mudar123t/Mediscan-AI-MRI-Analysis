from pathlib import Path
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")
PILOT_IMG = PROJECT_ROOT / "data" / "pilot" / "processed" / "images"
ROI_PRED  = PROJECT_ROOT / "data" / "pilot" / "roi_pred" / "images"
ROI_GT    = PROJECT_ROOT / "data" / "pilot" / "roi_gt" / "images"

def load_gray(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def main():
    split = "val"
    files = list((PILOT_IMG / split).glob("*.png"))
    f = random.choice(files)

    orig = load_gray(PILOT_IMG / split / f.name)
    roi_pred = load_gray(ROI_PRED / split / f.name)
    roi_gt = load_gray(ROI_GT / split / f.name)

    print("Sample:", f.name)
    print("orig min/max:", orig.min(), orig.max())
    print("roi_pred nonzero pixels:", int((roi_pred > 0).sum()))
    print("roi_gt nonzero pixels  :", int((roi_gt > 0).sum()))

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(orig, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(roi_gt, cmap="gray"); ax[1].set_title("ROI (GT mask)"); ax[1].axis("off")
    ax[2].imshow(roi_pred, cmap="gray"); ax[2].set_title("ROI (Pred mask)"); ax[2].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
