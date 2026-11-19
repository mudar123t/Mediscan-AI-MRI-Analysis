from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mediscan.data.dataset_cls import ClassificationDataset, CLASS_TO_IDX
from mediscan.classification.cnn import SimpleCNN

PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")
PILOT = PROJECT_ROOT / "data" / "pilot"
CKPT_DIR = PROJECT_ROOT / "outputs" / "pilot" / "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

def confusion_matrix(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def eval_one(name: str, test_dir: Path, ckpt_path: Path):
    ds = ClassificationDataset(test_dir)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, _ in dl:
            x = x.to(DEVICE)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred, num_classes=3)

    print(f"\n=== TEST: {name} ===")
    print("Accuracy:", round(float(acc), 4))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("Classes:", [IDX_TO_CLASS[i] for i in range(3)])

def main():
    full_test = PILOT / "processed" / "images" / "test"
    roi_gt_test = PILOT / "roi_gt" / "images" / "test"
    roi_pred_test = PILOT / "roi_pred" / "images" / "test"

    eval_one("full", full_test, CKPT_DIR / "cnn_full_best.pt")
    eval_one("roi_gt", roi_gt_test, CKPT_DIR / "cnn_roi_gt_best.pt")
    eval_one("roi_pred", roi_pred_test, CKPT_DIR / "cnn_roi_pred_best.pt")

if __name__ == "__main__":
    main()
