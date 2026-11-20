"""M2 demo logic: routing/referral decisions and preprocessing (no models needed)."""

import cv2
import numpy as np
import pytest

from mediscan.data.figshare import normalize_intensity, to_uint8
from mediscan.demo import decide, overlay_mask, overlay_uncertainty, preprocess_upload

PARAMS = {"score": "ens:disagree", "fill": 0.9, "tau1": 0.2, "temps": {"crop32": 1.0, "full": 1.0},
          "q": {"router@0.1": 0.6, "router@0.05": 0.9}}


def _rec(u, crop, full, area=100):
    r = {"ens:disagree": u, "pred_area": area}
    r.update({f"l_crop32_{c}": v for c, v in enumerate(crop)})
    r.update({f"l_full_{c}": v for c, v in enumerate(full)})
    return r


def test_routes_to_crop_when_u_at_or_below_tau1():
    d = decide(_rec(0.2, [5, 0, 0], [0, 5, 0]), PARAMS, 0.1)
    assert d["used_crop"] and d["pred"] == 0


def test_routes_to_full_when_u_above_tau1_or_missing():
    assert decide(_rec(0.21, [5, 0, 0], [0, 5, 0]), PARAMS, 0.1)["pred"] == 1
    d = decide(_rec(float("nan"), [5, 0, 0], [0, 5, 0], area=0), PARAMS, 0.1)  # empty mask -> fill (0.9) > tau1
    assert not d["used_crop"] and d["u"] == 0.9 and "no tumour region" in d["reason_route"]


def test_conformal_referral_cases():
    single = decide(_rec(0.0, [8, 0, 0], [0, 0, 0]), PARAMS, 0.1)          # p ~ 1 -> set {0}
    assert single["conformal_set"] == [0] and not single["refer"]
    ambiguous = decide(_rec(0.0, [3, 3, -5], [0, 0, 0]), PARAMS, 0.1)     # two classes ~0.5, 1 - p <= 0.6 -> both in set
    assert len(ambiguous["conformal_set"]) == 2 and ambiguous["refer"]
    empty = decide(_rec(0.0, [1, 0.9, 0.8], [0, 0, 0]), PARAMS, 0.1)      # all 1 - p > 0.6 -> empty set
    assert empty["conformal_set"] == [] and empty["refer"]
    assert decide(_rec(0.0, [1, 0.9, 0.8], [0, 0, 0]), PARAMS, 0.05)["refer"]  # alpha 0.05 uses its own q


def test_preprocess_matches_data_pipeline():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 4000, (512, 512)).astype(np.uint16)
    expected = to_uint8(cv2.resize(normalize_intensity(raw, 0.5, 99.5), (256, 256), interpolation=cv2.INTER_AREA))
    assert np.array_equal(preprocess_upload(raw), expected)
    rgb = np.repeat(rng.integers(0, 255, (300, 200, 1)).astype(np.uint8), 3, axis=2)
    assert preprocess_upload(rgb).shape == (256, 256)


@pytest.mark.parametrize("fn", [lambda x, m: overlay_mask(x, m), lambda x, m: overlay_uncertainty(x, m.astype(float))])
def test_overlays_are_rgb(fn):
    x = np.zeros((256, 256), np.uint8)
    m = np.zeros((256, 256), np.uint8)
    m[100:120, 100:120] = 1
    out = fn(x, m)
    assert out.shape == (256, 256, 3) and out.dtype == np.uint8
