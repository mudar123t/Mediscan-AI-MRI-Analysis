"""Tests for the environment and the reproducibility helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mediscan.segmentation.unet import UNet
from mediscan.utils.repro import create_run_dir, environment_info, load_config, seed_everything


def test_seed_everything_repeats_numpy_and_torch() -> None:
    seed_everything(123)
    a_np, a_t = np.random.rand(5), torch.rand(5)
    seed_everything(123)
    b_np, b_t = np.random.rand(5), torch.rand(5)
    assert np.allclose(a_np, b_np)
    assert torch.equal(a_t, b_t)


def test_environment_info_has_core_fields(repo_root: Path) -> None:
    info = environment_info(repo_root)
    for key in ("python", "torch", "numpy", "cuda_available", "git_commit"):
        assert key in info


def test_create_run_dir_writes_config_and_env(tmp_path: Path) -> None:
    cfg = {"experiment": "unit-test", "seed": 0}
    run_dir = create_run_dir(tmp_path, "run1", cfg)
    assert load_config(run_dir / "config.yaml") == cfg
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert "torch" in env
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "run1", cfg)


def test_all_configs_parse(repo_root: Path) -> None:
    configs = sorted((repo_root / "configs").glob("*.yaml"))
    assert configs, "expected at least one config file"
    for path in configs:
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict), path


def test_semester1_unet_still_builds() -> None:
    """The semester-1 U-Net is kept as the fair baseline (plan §3.2, EXP-002a)."""
    model = UNet(in_channels=1, out_channels=1)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 7_762_465  # value recorded in EXP-000
    out = model(torch.zeros(1, 1, 64, 64))
    assert out.shape == (1, 1, 64, 64)


@pytest.mark.gpu
def test_cuda_forward_pass_with_amp() -> None:
    model = UNet(in_channels=1, out_channels=1).cuda()
    x = torch.zeros(2, 1, 256, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
    assert out.shape == (2, 1, 256, 256)
