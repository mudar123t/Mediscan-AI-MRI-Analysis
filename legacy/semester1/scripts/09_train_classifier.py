from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mediscan.data.dataset_cls import ClassificationDataset
from mediscan.classification.cnn import SimpleCNN
from mediscan.classification.metrics import accuracy

PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")
PILOT = PROJECT_ROOT / "data" / "pilot"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-3

def run_epoch(model, loader, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
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
            total_acc += accuracy(logits, y) * bs
            n += bs

    return total_loss / n, total_acc / n

def train_one(experiment_name: str, train_dir: Path, val_dir: Path):
    out_dir = PROJECT_ROOT / "outputs" / "pilot" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"cnn_{experiment_name}_best.pt"

    train_ds = ClassificationDataset(train_dir)
    val_ds = ClassificationDataset(val_dir)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(num_classes=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0

    print(f"\n=== Train CNN: {experiment_name} ===")
    print("Device:", DEVICE)
    print("Train size:", len(train_ds), "| Val size:", len(val_ds))

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, loss_fn, optimizer=opt)
        va_loss, va_acc = run_epoch(model, val_dl, loss_fn)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f} || val loss={va_loss:.4f} acc={va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✅ Saved best: {ckpt_path} (val acc={best_val_acc:.3f})")

    return ckpt_path

def main():
    # 1) Full image baseline
    full_train = PILOT / "processed" / "images" / "train"
    full_val   = PILOT / "processed" / "images" / "val"

    # 2) ROI from GT masks
    roi_gt_train = PILOT / "roi_gt" / "images" / "train"
    roi_gt_val   = PILOT / "roi_gt" / "images" / "val"

    # 3) ROI from predicted masks
    roi_pred_train = PILOT / "roi_pred" / "images" / "train"
    roi_pred_val   = PILOT / "roi_pred" / "images" / "val"

    train_one("full", full_train, full_val)
    train_one("roi_gt", roi_gt_train, roi_gt_val)
    train_one("roi_pred", roi_pred_train, roi_pred_val)

    print("\nDone. Next: scripts/10_eval_classifier.py")

if __name__ == "__main__":
    main()
