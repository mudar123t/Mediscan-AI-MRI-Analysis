"""PyTorch datasets over the processed Figshare arrays.

Rows come only from split files (never from globbing directories), which
removes the semester-1 stale-file contamination by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FigshareArrays:
    """The processed dataset held in RAM (about 0.4 GB for 256²)."""

    images: np.ndarray  # uint8 (N, H, W)
    masks: np.ndarray  # uint8 (N, H, W), {0, 1}
    meta: pd.DataFrame

    @classmethod
    def load(cls, processed_dir: str | Path, mmap: bool = False) -> "FigshareArrays":
        d = Path(processed_dir)
        mode = "r" if mmap else None
        images = np.load(d / "images.npy", mmap_mode=mode)
        masks = np.load(d / "masks.npy", mmap_mode=mode)
        meta = pd.read_csv(d / "metadata.csv", dtype={"pid": str})
        if not (len(images) == len(masks) == len(meta)):
            raise ValueError("images, masks and metadata have different lengths")
        return cls(images=images, masks=masks, meta=meta)


class FigshareSegDataset(Dataset):
    """(image, mask) pairs for segmentation. Image in [0, 1], shape (1, H, W)."""

    def __init__(
        self, arrays: FigshareArrays, rows: Sequence[int], transform: A.Compose | None = None
    ) -> None:
        self.arrays = arrays
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        image = np.asarray(self.arrays.images[row])
        mask = np.asarray(self.arrays.masks[row])
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return {
            "image": torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0),
            "mask": torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            "label": int(self.arrays.meta.at[row, "class_index"]),
            "row": row,
        }


class FigshareClsDataset(Dataset):
    """Classification samples. ``view_fn(image, mask) -> image`` builds the input view.

    ``mask_source`` is ``"gt"`` (ground-truth masks) or an array of predicted
    masks aligned with the dataset rows. The view functions themselves are
    implemented in M0.4 (``mediscan.views``).
    """

    def __init__(
        self,
        arrays: FigshareArrays,
        rows: Sequence[int],
        view_fn: Any = None,
        mask_source: str | np.ndarray = "gt",
        transform: A.Compose | None = None,
    ) -> None:
        self.arrays = arrays
        self.rows = list(rows)
        self.view_fn = view_fn
        self.masks = arrays.masks if isinstance(mask_source, str) and mask_source == "gt" else mask_source
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        image = np.asarray(self.arrays.images[row])
        mask = np.asarray(self.masks[row])
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        if self.view_fn is not None:
            image = self.view_fn(image, mask)
        return {
            "image": torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0),
            "label": int(self.arrays.meta.at[row, "class_index"]),
            "row": row,
        }


def load_predicted_masks(
    seg_fold_dir: str | Path, n_rows: int, threshold: float = 0.5, keep_largest: bool = True
) -> tuple[np.ndarray, list[int]]:
    """Binary predicted masks of one segmentation fold run, aligned to dataset rows.

    Reads ``pred_val.npz`` and ``pred_test.npz`` (sigmoid probabilities stored as
    uint8 0..255) written by ``scripts/train_seg.py``. Rows the segmenter did not
    predict stay all-zero; the returned list names the rows that were filled.
    """
    from mediscan.metrics.seg import largest_component

    d = Path(seg_fold_dir)
    masks: np.ndarray | None = None
    filled: list[int] = []
    for part in ("val", "test"):
        z = np.load(d / f"pred_{part}.npz")
        rows, probs = z["rows"], z["probs"]
        if masks is None:
            masks = np.zeros((n_rows, *probs.shape[1:]), dtype=np.uint8)
        for r, p in zip(rows, probs):
            m = (p.astype(np.float32) / 255.0 > threshold).astype(np.uint8)
            masks[int(r)] = largest_component(m) if keep_largest else m
        filled.extend(int(r) for r in rows)
    assert masks is not None
    return masks, filled
