from pathlib import Path
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from mediscan.segmentation.unet import UNet
from mediscan.data.dataset_seg import SegmentationDataset

PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")
PILOT_ROOT = PROJECT_ROOT / "data" / "pilot"
CKPT = PROJECT_ROOT / "outputs" / "pilot" / "checkpoints" / "unet_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ds = SegmentationDataset(PILOT_ROOT, "val")
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    idxs = random.sample(range(len(ds)), k=min(6, len(ds)))

    for idx in idxs:
        x, y, name = ds[idx]
        x = x.unsqueeze(0).to(DEVICE)  # [1,1,H,W]

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0,0]
            pred = (probs > 0.5).astype(np.float32)

        img = (x.cpu().numpy()[0,0] * 255).astype(np.uint8)
        gt = y.numpy()[0]

        print(f"\n{name}")
        print("GT tumor pixels:", int(gt.sum()), " / ", gt.size)
        print("Pred tumor pixels:", int(pred.sum()), " / ", pred.size)

        # overlay (red)
        img01 = img.astype(np.float32) / 255.0
        overlay = np.stack([img01, img01, img01], axis=-1)
        overlay[..., 0] = np.maximum(overlay[..., 0], pred)

        fig, ax = plt.subplots(1,4, figsize=(14,4))
        ax[0].imshow(img01, cmap="gray"); ax[0].set_title("MRI"); ax[0].axis("off")
        ax[1].imshow(gt, cmap="gray"); ax[1].set_title("GT mask"); ax[1].axis("off")
        ax[2].imshow(probs, cmap="gray"); ax[2].set_title("Pred prob"); ax[2].axis("off")
        ax[3].imshow(overlay); ax[3].set_title("Pred overlay"); ax[3].axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
