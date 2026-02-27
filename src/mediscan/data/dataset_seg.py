from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Loads preprocessed pilot data:
      images: uint8 PNG 0..255
      masks : uint8 PNG {0,255}
    Returns:
      image: float32 tensor [1, H, W] in [0,1]
      mask : float32 tensor [1, H, W] in {0,1}
    """

    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        self.split = split

        self.img_dir = self.root / "processed" / "images" / split
        self.mask_dir = self.root / "processed" / "masks" / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing masks dir: {self.mask_dir}")

        self.items = sorted([p.name for p in self.img_dir.glob("*.png")])
        if len(self.items) == 0:
            raise ValueError(f"No PNG files found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        name = self.items[idx]
        img_path = self.img_dir / name
        mask_path = self.mask_dir / name

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"Read failed for {img_path} or {mask_path}")

        # to float32 [0,1]
        img = img.astype(np.float32) / 255.0

        # mask to {0,1}
        mask = (mask > 0).astype(np.float32)

        # add channel dim: [1,H,W]
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return img_t, mask_t, name
