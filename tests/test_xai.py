"""EXP-010 Grad-CAM helpers."""

import numpy as np
import torch

from mediscan.xai import backproject, gradcam, region_fractions


def test_region_fractions_sum_to_one_and_locate_mass():
    cam = np.zeros((64, 64), np.float32)
    cam[10:20, 10:20] = 1.0                       # all mass inside the region
    reg = np.zeros((64, 64), np.uint8)
    reg[5:25, 5:25] = 1
    f_in, f_ring, f_out = region_fractions(cam, reg, ring_px=4)
    assert np.isclose(f_in, 1.0) and np.isclose(f_ring, 0.0) and np.isclose(f_out, 0.0)
    cam2 = np.zeros_like(cam)
    cam2[27, 15] = 1.0                            # 2 px below the region -> ring
    cam2[60, 60] = 1.0                            # far away -> out
    f = region_fractions(cam2, reg, ring_px=4)
    assert np.allclose(f, (0.0, 0.5, 0.5))


def test_region_fractions_nan_for_empty_map_or_region():
    assert np.isnan(region_fractions(np.zeros((8, 8)), np.ones((8, 8)))[0])
    assert np.isnan(region_fractions(np.ones((8, 8)), np.zeros((8, 8)))[0])


def test_backproject_places_crop_inside_box():
    out = backproject(np.ones((256, 256), np.float32), (10, 20, 74, 84), (256, 256))
    assert out[10:74, 20:84].min() > 0.99 and out.sum() == out[10:74, 20:84].sum()
    full = np.random.default_rng(0).random((256, 256)).astype(np.float32)
    assert np.array_equal(backproject(full, None, (256, 256)), full)


def test_gradcam_highlights_the_informative_region():
    torch.manual_seed(0)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = torch.nn.Conv2d(1, 1, 1, bias=False)
            self.feat.weight.data.fill_(1.0)
            self.fc = torch.nn.Linear(1, 2, bias=False)
            self.fc.weight.data = torch.tensor([[1.0], [-1.0]])

        def forward(self, x):
            return self.fc(self.feat(x).mean(dim=(2, 3)))

    net = Net()
    x = torch.zeros(1, 1, 16, 16)
    x[0, 0, 2:6, 2:6] = 1.0                        # class 0 evidence only in the top-left
    cam, logits = gradcam(net, net.feat, x)
    assert logits.argmax() == 0 and cam.shape == (1, 16, 16)
    assert cam[0, 2:6, 2:6].sum() > 0.99 * cam.sum()
