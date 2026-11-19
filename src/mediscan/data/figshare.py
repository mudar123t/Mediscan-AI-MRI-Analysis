"""Reader and converter for the Figshare brain tumor dataset (Cheng et al.).

Dataset: https://doi.org/10.6084/m9.figshare.1512427 (CC BY 4.0).
Each ``<idx>.mat`` file (MATLAB v7.3 / HDF5) holds a struct ``cjdata`` with
``label`` (1 meningioma, 2 glioma, 3 pituitary), ``PID``, ``image``,
``tumorBorder`` ([x1, y1, x2, y2, ...], 1-based planar coordinates) and
``tumorMask``.

Orientation: HDF5 stores MATLAB arrays column-major, so h5py returns the
*transpose* of the MATLAB array. We transpose ``image`` and ``tumorMask``
back. EXP-001 verifies this against the hand-drawn ``tumorBorder`` polygon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np

#: Figshare label (1..3) -> class index (0..2) and name.
LABEL_TO_INDEX: dict[int, int] = {1: 0, 2: 1, 3: 2}
CLASS_NAMES: tuple[str, ...] = ("meningioma", "glioma", "pituitary")


@dataclass(frozen=True)
class FigshareCase:
    """One slice read from a ``.mat`` file, in MATLAB (row, column) orientation."""

    index: int  # file number, 1..3064
    pid: str
    label: int  # 1..3 as stored in the file
    image: np.ndarray  # int16 / float, (H, W)
    mask: np.ndarray  # uint8 {0, 1}, (H, W)
    border_xy: np.ndarray  # float64, (K, 2), 1-based (x = column, y = row)

    @property
    def class_index(self) -> int:
        return LABEL_TO_INDEX[self.label]


def _decode_pid(arr: np.ndarray) -> str:
    return "".join(chr(int(c)) for c in arr.flatten()).strip()


def read_case(path: str | Path) -> FigshareCase:
    """Read one Figshare ``.mat`` file and return it in MATLAB orientation."""
    path = Path(path)
    with h5py.File(path, "r") as fh:
        cj = fh["cjdata"]
        image = np.asarray(cj["image"][()]).T
        mask = np.asarray(cj["tumorMask"][()]).T.astype(np.uint8)
        label = int(np.asarray(cj["label"][()]).item())
        pid = _decode_pid(cj["PID"][()])
        border = np.asarray(cj["tumorBorder"][()], dtype=np.float64).flatten()
    if label not in LABEL_TO_INDEX:
        raise ValueError(f"{path}: unexpected label {label}")
    if border.size % 2:
        raise ValueError(f"{path}: tumorBorder has an odd number of values")
    return FigshareCase(
        index=int(path.stem),
        pid=pid,
        label=label,
        image=image,
        mask=mask,
        border_xy=border.reshape(-1, 2),
    )


def border_to_mask(border_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise the 1-based (x, y) border polygon into a {0, 1} mask."""
    pts = np.round(border_xy - 1.0).astype(np.int32)
    out = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(out, [pts], 1)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary masks (1.0 if both are empty)."""
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def normalize_intensity(image: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Clip to per-image percentiles and scale to [0, 1] (float32).

    Percentile clipping (rather than plain min-max) limits the effect of a
    few very bright pixels, which vary a lot between scanners.
    """
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(img)
    lo, hi = np.float32(lo), np.float32(hi)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resize_pair(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize image with area averaging and mask with nearest neighbour.

    Nearest neighbour keeps the mask binary (semester 1 interpolated masks).
    Images that already have the target size are returned unchanged.
    """
    if image.shape == (size, size):
        return image, mask
    img = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, msk


def to_uint8(image01: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] float image to uint8 with rounding."""
    return np.round(np.clip(image01, 0.0, 1.0) * 255.0).astype(np.uint8)


def read_folds(cvind_path: str | Path) -> np.ndarray:
    """Return the fold number (1..5) of each slice, ordered by file number 1..N."""
    with h5py.File(cvind_path, "r") as fh:
        folds = np.asarray(fh["cvind"][()]).flatten()
    if not np.all(np.isin(folds, [1, 2, 3, 4, 5])):
        raise ValueError("cvind.mat contains values outside 1..5")
    return folds.astype(np.int64)
