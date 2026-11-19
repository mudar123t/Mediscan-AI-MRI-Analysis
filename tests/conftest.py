"""Shared pytest fixtures and skip rules."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = REPO_ROOT / "dataset" / "dataset"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``data`` tests without the raw dataset and ``gpu`` tests without CUDA."""
    has_data = (RAW_DATA_DIR / "cvind.mat").is_file()
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    skip_data = pytest.mark.skip(reason="raw Figshare data not found in dataset/dataset")
    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    for item in items:
        if "data" in item.keywords and not has_data:
            item.add_marker(skip_data)
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
