"""Small, explicit reproducibility helpers for TMF experiments."""
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without changing the experiment protocol."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve a requested device and fail clearly when CUDA was requested but is absent."""
    value = str(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {value!r}, but CUDA is not available")
    return torch.device(value)


def git_commit(project_root: Path) -> str:
    """Return the current code commit, or ``unknown`` outside a Git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def worker_seed(worker_id: int) -> int:
    """Return the PyTorch-derived seed for a DataLoader worker."""
    del worker_id
    return int(torch.initial_seed() % (2**32))


def seed_worker(worker_id: int) -> None:
    seed = worker_seed(worker_id)
    random.seed(seed)
    np.random.seed(seed)
