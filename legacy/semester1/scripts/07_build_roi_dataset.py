from pathlib import Path
import numpy as np
import cv2
import torch

from mediscan.segmentation.unet import UNet

PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")
PILOT_ROOT = PROJECT_ROOT / "data" / "pilot"

CKPT = PROJECT_ROOT / "outputs" / "pilot" / "checkpoints" / "unet_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Where to save ROI datasets
OUT_ROI_PRED = PROJECT_ROOT / "data" / "pilot" / "roi_pred"   # ROI using predicted masks
OUT_ROI_GT   = PROJECT_ROOT / "data" / "pilot" / "roi_gt"     # ROI using ground-truth masks

THRESH = 0.3  # lower threshold to reduce false negatives


def ensure_dirs(base: Path):
    for split in ["train", "val", "test"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img


def keep_largest_component(mask_u8: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    This removes tiny scattered dots and keeps the main blob.
    """
    m = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    # num includes background label 0
    if num <= 1:
        return mask_u8  # empty mask

    # pick largest component excluding background
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (labels == largest).astype(np.uint8) * 255
    return out


def predict_mask(model, img_u8: np.ndarray) -> np.ndarray:
    """
    img_u8: [H,W] uint8
    returns: [H,W] uint8 mask {0,255}
    """
    x = (img_u8.astype(np.float32) / 255.0)[None, None, ...]  # [1,1,H,W]
    x = torch.from_numpy(x).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

    pred = (probs > THRESH).astype(np.uint8) * 255

    # ✅ IMPORTANT: clean the mask here
    pred = keep_largest_component(pred)

    return pred


def apply_roi(img_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    ROI = image * (mask>0)
    """
    roi = img_u8.copy()
    roi[mask_u8 == 0] = 0
    return roi


def process_split(split: str, model):
    img_dir = PILOT_ROOT / "processed" / "images" / split
    gt_mask_dir = PILOT_ROOT / "processed" / "masks" / split

    names = sorted([p.name for p in img_dir.glob("*.png")])
    for name in names:
        img_path = img_dir / name
        gt_mask_path = gt_mask_dir / name

        img = load_gray(img_path)
        gt_mask = load_gray(gt_mask_path)

        # predicted mask (threshold + keep largest component)
        pred_mask = predict_mask(model, img)

        # ROI datasets
        roi_pred = apply_roi(img, pred_mask)
        roi_gt = apply_roi(img, gt_mask)

        cv2.imwrite(str(OUT_ROI_PRED / "images" / split / name), roi_pred)
        cv2.imwrite(str(OUT_ROI_GT / "images" / split / name), roi_gt)

    print(f"✅ Built ROI for split '{split}' | count={len(names)}")


def main():
    print("=== Step 4: Build ROI datasets (Pred + GT) ===")
    print("Device:", DEVICE)
    print("Threshold:", THRESH)

    # create dirs
    ensure_dirs(OUT_ROI_PRED)
    ensure_dirs(OUT_ROI_GT)

    # load model
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    for split in ["train", "val", "test"]:
        process_split(split, model)

    print("\nSaved ROI datasets:")
    print(" - ROI from predicted masks:", OUT_ROI_PRED)
    print(" - ROI from GT masks      :", OUT_ROI_GT)
    print("Next: run scripts/08_check_roi_dataset.py")


if __name__ == "__main__":
    main()
