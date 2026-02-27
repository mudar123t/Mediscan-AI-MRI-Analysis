from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

CLASS_TO_IDX = {"glioma": 0, "meningioma": 1, "pituitary": 2}

def parse_label_from_name(filename: str) -> int:
    # expected: glioma_123.png
    prefix = filename.split("_")[0]
    if prefix not in CLASS_TO_IDX:
        raise ValueError(f"Cannot parse class from {filename}")
    return CLASS_TO_IDX[prefix]

class ClassificationDataset(Dataset):
    """
    Works with:
      - full images: data/pilot/processed/images/{split}
      - ROI images:  data/pilot/roi_gt/images/{split} or roi_pred/...

    Returns:
      x: float32 tensor [1,256,256] in [0,1]
      y: int64 label {0,1,2}
    """

    def __init__(self, img_dir: Path):
        self.img_dir = Path(img_dir)
        self.items = sorted([p.name for p in self.img_dir.glob("*.png")])
        if len(self.items) == 0:
            raise ValueError(f"No PNG images found in {self.img_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        name = self.items[idx]
        path = self.img_dir / name

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)

        x = (img.astype(np.float32) / 255.0)
        x = torch.from_numpy(x).unsqueeze(0)  # [1,H,W]

        y = parse_label_from_name(name)
        y = torch.tensor(y, dtype=torch.long)
        return x, y, name
