"""Reproducibility helpers: seeding, environment capture and run folders.

Every training or evaluation run calls :func:`seed_everything` and
:func:`create_run_dir`, which stores the config and the environment next to
the run outputs.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    Args:
        seed: The seed value.
        deterministic: If True, request deterministic cuDNN kernels
            (slower, but repeatable).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a core dependency
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Collect versions, hardware and git state for the run record."""
    repo_root = repo_root or Path.cwd()
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "git_commit": _git(["rev-parse", "HEAD"], repo_root),
        "git_branch": _git(["branch", "--show-current"], repo_root),
        "git_dirty": bool(_git(["status", "--porcelain"], repo_root)),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_mib"] = props.total_memory // (1024 * 1024)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
    except ImportError:  # pragma: no cover
        info["torch"] = None
    for pkg in ("torchvision", "segmentation_models_pytorch", "timm", "albumentations", "sklearn"):
        try:
            module = __import__(pkg)
            info[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[pkg] = None
    return info


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level.")
    return cfg


def create_run_dir(
    root: str | Path, name: str, config: dict[str, Any], exist_ok: bool = False
) -> Path:
    """Create ``root/name`` and write ``config.yaml`` and ``environment.json`` into it.

    A fresh folder per run avoids the semester-1 bug where stale files from
    earlier runs were mixed into evaluation directories.

    Raises:
        FileExistsError: If the folder exists and ``exist_ok`` is False.
    """
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    with open(run_dir / "environment.json", "w", encoding="utf-8") as fh:
        json.dump(environment_info(), fh, indent=2)
    return run_dir
