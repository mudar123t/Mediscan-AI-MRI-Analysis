"""Patient-disjoint train/val/test splits from the Figshare ``cvind.mat`` folds.

For outer test fold ``k`` (0-based, 0..4): val is fold ``(k + 1) % 5``, and
train is the other three folds (about 60/20/20 by patient). Folds are
0-based here; ``cvind.mat`` stores them as 1..5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

N_FOLDS = 5


@dataclass(frozen=True)
class Split:
    """Row indices (into ``metadata.csv``) for one outer fold."""

    test_fold: int
    val_fold: int
    train: list[int]
    val: list[int]
    test: list[int]


class LeakageError(AssertionError):
    """Raised when a patient appears in more than one split part."""


def make_split(meta: pd.DataFrame, test_fold: int, n_folds: int = N_FOLDS) -> Split:
    """Build the split for one outer test fold from ``meta['fold']`` (0-based)."""
    if not 0 <= test_fold < n_folds:
        raise ValueError(f"test_fold must be in 0..{n_folds - 1}")
    val_fold = (test_fold + 1) % n_folds
    fold = meta["fold"].to_numpy()
    rows = meta.index.to_numpy()
    split = Split(
        test_fold=test_fold,
        val_fold=val_fold,
        train=sorted(int(i) for i in rows[(fold != test_fold) & (fold != val_fold)]),
        val=sorted(int(i) for i in rows[fold == val_fold]),
        test=sorted(int(i) for i in rows[fold == test_fold]),
    )
    assert_patient_disjoint(meta, split)
    return split


def assert_patient_disjoint(meta: pd.DataFrame, split: Split) -> None:
    """Raise :class:`LeakageError` if any patient is in two parts, or rows overlap."""
    parts = {"train": split.train, "val": split.val, "test": split.test}
    pids = {name: set(meta.loc[idx, "pid"]) for name, idx in parts.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = pids[a] & pids[b]
        if shared:
            raise LeakageError(f"fold {split.test_fold}: {len(shared)} patients in both {a} and {b}")
        if set(parts[a]) & set(parts[b]):
            raise LeakageError(f"fold {split.test_fold}: rows shared between {a} and {b}")
    if sum(len(v) for v in parts.values()) != len(meta):
        raise LeakageError(f"fold {split.test_fold}: split does not cover all rows exactly once")


def save_split(split: Split, path: str | Path) -> None:
    """Write a split as JSON (file numbers are recoverable via metadata.csv)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(split), indent=1), encoding="utf-8")


def load_split(path: str | Path, meta: pd.DataFrame | None = None) -> Split:
    """Read a split JSON; if ``meta`` is given, re-check patient disjointness."""
    split = Split(**json.loads(Path(path).read_text(encoding="utf-8")))
    if meta is not None:
        assert_patient_disjoint(meta, split)
    return split
