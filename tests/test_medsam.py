"""EXP-011 MedSAM helpers (no model weights needed)."""

import numpy as np

from mediscan.medsam import box_from_mask, jitter_boxes, signals, to_pixel_values


def test_pixel_values_shape_and_normalisation():
    img = np.zeros((256, 256), np.uint8)
    img[100:150, 100:150] = 200
    x = to_pixel_values(img)
    assert x.shape == (3, 1024, 1024) and x.dtype == np.float32
    assert np.isclose(x[0].min(), (0 - 0.485) / 0.229, atol=0.05)  # background maps to (0 - mean) / std


def test_box_from_mask_is_end_exclusive_and_scaled():
    m = np.zeros((256, 256), np.uint8)
    m[10:20, 30:41] = 1
    assert np.allclose(box_from_mask(m), np.array([30, 10, 41, 20]) * 4)
    assert box_from_mask(np.zeros((8, 8))) is None


def test_jitter_is_deterministic_bounded_and_valid():
    box = np.array([100, 200, 300, 260], np.float32)
    a = jitter_boxes(box, np.random.default_rng([0, 5, 1]))
    b = jitter_boxes(box, np.random.default_rng([0, 5, 1]))
    assert np.array_equal(a, b) and a.shape == (8, 4)
    assert (np.abs(a - box) <= np.array([20, 6, 20, 6]) + 1e-4).all()  # 10 % of width 200 / height 60
    assert (a[:, 2] > a[:, 0]).all() and (a[:, 3] > a[:, 1]).all() and a.min() >= 0 and a.max() <= 1024


def test_signals_zero_when_medsam_agrees_and_one_when_disjoint():
    ref = np.zeros((32, 32), np.uint8)
    ref[5:15, 5:15] = 1
    same = np.repeat(ref[None], 9, axis=0)
    assert all(np.isclose(v, 0.0) for v in signals(ref, same).values())
    other = np.zeros_like(same)
    other[:, 20:30, 20:30] = 1
    s = signals(ref, other)
    assert np.isclose(s["sam:disagree"], 1.0) and np.isclose(s["sam:jitter"], 1.0) and np.isclose(s["sam:selfvar"], 0.0)
