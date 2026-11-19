"""Tests for the Figshare reader, splits (leakage guards) and datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mediscan.data import figshare as fs
from mediscan.data.augment import train_transform
from mediscan.data.datasets import FigshareArrays, FigshareSegDataset
from mediscan.data.splits import LeakageError, Split, assert_patient_disjoint, load_split, make_split

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "figshare_256"
SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


# ---------- synthetic (always run) ----------
def _toy_meta() -> pd.DataFrame:
    # 10 patients, 2 slices each, patient p in fold p % 5
    rows = [{"pid": f"p{p}", "fold": p % 5, "class_index": p % 3} for p in range(10) for _ in range(2)]
    return pd.DataFrame(rows)


def test_make_split_rule_and_disjointness() -> None:
    meta = _toy_meta()
    for k in range(5):
        s = make_split(meta, k)
        assert s.val_fold == (k + 1) % 5
        assert set(meta.loc[s.test, "fold"]) == {k}
        assert set(meta.loc[s.val, "fold"]) == {(k + 1) % 5}
        assert len(s.train) + len(s.val) + len(s.test) == len(meta)


def test_leakage_is_detected() -> None:
    meta = _toy_meta()
    good = make_split(meta, 0)
    leaky = Split(0, 1, good.train[:-1], good.val, good.test + [good.train[-1]])
    # moving one slice of a training patient into test puts that patient in both parts
    with pytest.raises(LeakageError):
        assert_patient_disjoint(meta, leaky)


def test_border_mask_and_iou() -> None:
    square = np.array([[3, 3], [8, 3], [8, 8], [3, 8]], dtype=float)  # 1-based x, y
    m = fs.border_to_mask(square, (12, 12))
    assert m[2:8, 2:8].all() and m.sum() == 36
    assert fs.iou(m, m) == 1.0
    assert fs.iou(m, m.T) == 1.0  # symmetric square
    assert fs.iou(np.zeros((4, 4)), np.zeros((4, 4))) == 1.0


def test_resize_pair_keeps_mask_binary() -> None:
    rng = np.random.default_rng(0)
    img = rng.random((512, 512)).astype(np.float32)
    mask = (rng.random((512, 512)) > 0.7).astype(np.uint8)
    img_r, mask_r = fs.resize_pair(img, mask, 256)
    assert img_r.shape == mask_r.shape == (256, 256)
    assert set(np.unique(mask_r)) <= {0, 1}


def test_normalize_intensity_range() -> None:
    img = np.arange(10_000, dtype=np.int16).reshape(100, 100)
    out = fs.normalize_intensity(img)
    assert out.dtype == np.float32 and out.min() == 0.0 and out.max() == 1.0


def test_augmentation_moves_mask_with_image() -> None:
    img = np.zeros((64, 64), np.uint8)
    img[10:20, 30:40] = 255
    mask = (img > 0).astype(np.uint8)
    tf = train_transform("default")
    for seed in range(10):
        tf.set_random_seed(seed)
        out = tf(image=img, mask=mask)
        assert set(np.unique(out["mask"])) <= {0, 1}
        bright = out["image"] > 128
        if out["mask"].sum() > 20:
            assert (bright & (out["mask"] > 0)).sum() / max(bright.sum(), 1) > 0.8


# ---------- real data (skipped without the processed set) ----------
needs_processed = pytest.mark.skipif(not (PROCESSED / "metadata.csv").is_file(), reason="run scripts/prepare_figshare.py first")


@needs_processed
def test_processed_dataset_facts() -> None:
    arr = FigshareArrays.load(PROCESSED, mmap=True)
    meta = arr.meta
    assert len(meta) == 3064 and meta["pid"].nunique() == 233
    assert meta["class_name"].value_counts().to_dict() == {"glioma": 1426, "pituitary": 930, "meningioma": 708}
    assert (meta.groupby("pid")["fold"].nunique() == 1).all()
    assert (meta["mask_area"] > 0).all()
    assert meta["border_iou"].min() > 0.8  # orientation check (EXP-001)
    assert arr.images.shape == (3064, 256, 256) and arr.images.dtype == np.uint8


@needs_processed
@pytest.mark.parametrize("k", range(5))
def test_tracked_split_files_are_patient_disjoint(k: int) -> None:
    meta = FigshareArrays.load(PROCESSED, mmap=True).meta
    split = load_split(SPLITS / f"figshare_fold{k}.json", meta)  # re-checks disjointness
    assert split == make_split(meta, k)


@needs_processed
def test_seg_dataset_item() -> None:
    arr = FigshareArrays.load(PROCESSED, mmap=True)
    ds = FigshareSegDataset(arr, [0, 1, 2], transform=train_transform("default"))
    item = ds[0]
    assert item["image"].shape == (1, 256, 256) and item["mask"].shape == (1, 256, 256)
    assert 0.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
    assert item["label"] in (0, 1, 2)
