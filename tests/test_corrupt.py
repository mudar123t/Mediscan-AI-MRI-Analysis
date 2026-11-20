"""Tests for synthetic shifts (EXP-008)."""

from __future__ import annotations

import numpy as np
import pytest

from mediscan.corrupt import KINDS, corrupt


@pytest.mark.parametrize("kind", KINDS)
def test_corrupt_changes_more_with_severity_and_is_deterministic(kind: str) -> None:
    rng0 = np.random.default_rng(0)
    img = (np.clip(rng0.normal(0.4, 0.15, (64, 64)), 0, 1) * 255).astype(np.uint8)
    diffs = []
    for sev in (1, 2, 3):
        a = corrupt(img, kind, sev, np.random.default_rng(5))
        b = corrupt(img, kind, sev, np.random.default_rng(5))
        assert a.dtype == np.uint8 and a.shape == img.shape and (a == b).all()
        diffs.append(np.abs(a.astype(float) - img).mean())
    assert diffs[0] < diffs[1] < diffs[2]
