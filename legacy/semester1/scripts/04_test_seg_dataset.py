from pathlib import Path
import torch
from torch.utils.data import DataLoader
from mediscan.data.dataset_seg import SegmentationDataset

PROJECT_ROOT = Path(r"C:\Users\shawa\Desktop\final project")
PILOT_ROOT = PROJECT_ROOT / "data" / "pilot"

def main():
    ds = SegmentationDataset(PILOT_ROOT, split="train")
    print("Train size:", len(ds))

    x, y, name = ds[0]
    print("Sample name:", name)
    print("Image tensor:", x.shape, x.dtype, "min/max:", float(x.min()), float(x.max()))
    print("Mask tensor :", y.shape, y.dtype, "unique:", torch.unique(y))

    dl = DataLoader(ds, batch_size=4, shuffle=True)
    xb, yb, names = next(iter(dl))
    print("Batch image:", xb.shape)
    print("Batch mask :", yb.shape)
    print("Names:", names[:4])

if __name__ == "__main__":
    main()
