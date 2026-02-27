from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mediscan.data.dataset_seg import SegmentationDataset
from mediscan.segmentation.unet import UNet
from mediscan.segmentation.metrics import dice_score, iou_score
from mediscan.segmentation.losses import BCEDiceLoss

PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")
PILOT_ROOT = PROJECT_ROOT / "data" / "pilot"
OUT_DIR = PROJECT_ROOT / "outputs" / "pilot" / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-3

def run_epoch(model, loader, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss = loss_fn(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_dice += float(dice_score(logits, y).item()) * bs
            total_iou += float(iou_score(logits, y).item()) * bs
            n += bs

    return total_loss / n, total_dice / n, total_iou / n

def main():
    print("=== Step 3: Train U-Net (Pilot) ===")
    print("Device:", DEVICE)

    train_ds = SegmentationDataset(PILOT_ROOT, "train")
    val_ds = SegmentationDataset(PILOT_ROOT, "val")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = BCEDiceLoss(bce_weight=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = -1.0
    best_path = OUT_DIR / "unet_best.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_dice, tr_iou = run_epoch(model, train_dl, loss_fn, optimizer=opt)
        va_loss, va_dice, va_iou = run_epoch(model, val_dl, loss_fn)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={tr_loss:.4f} dice={tr_dice:.4f} iou={tr_iou:.4f} || "
            f"val loss={va_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f}"
        )

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved best model: {best_path} (val dice={best_dice:.4f})")

    print("Done. Best val dice:", best_dice)

if __name__ == "__main__":
    main()
