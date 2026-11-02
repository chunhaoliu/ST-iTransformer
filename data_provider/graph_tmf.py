"""Validated graph TMF NPZ schema, windows, and deterministic missingness."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .time_features import validate_time_features


class DatasetValidationError(ValueError):
    """Raised when a graph TMF NPZ violates its data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GraphTMFData:
    x: np.ndarray
    adj: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    dates: np.ndarray
    names: np.ndarray
    target_columns: np.ndarray
    target: str
    train_end: int
    val_end: int
    source_path: Path
    fingerprint: str

    @property
    def n_timesteps(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.x.shape[2])


def _scalar(payload: object, default: str = "") -> str:
    array = np.asarray(payload)
    return str(array.item()) if array.ndim == 0 else str(array.reshape(-1)[0])


def validate_graph_tmf(data: GraphTMFData) -> GraphTMFData:
    x = np.asarray(data.x)
    if x.ndim != 3:
        raise DatasetValidationError(f"x must have shape [T,N,C], got {x.shape}")
    timesteps, nodes, channels = x.shape
    if timesteps <= 0 or nodes <= 0 or channels <= 0:
        raise DatasetValidationError("x must have positive dimensions")
    if not np.isfinite(x).all():
        raise DatasetValidationError("x contains non-finite values")
    adj = np.asarray(data.adj)
    if adj.shape != (nodes, nodes):
        raise DatasetValidationError(f"adj must have shape {(nodes, nodes)}, got {adj.shape}")
    if not np.isfinite(adj).all() or (adj < 0).any():
        raise DatasetValidationError("adj must be finite and non-negative")
    if not np.allclose(adj, adj.T):
        raise DatasetValidationError("adjacency must be symmetric for the coordinate mask protocol")
    mean = np.asarray(data.mean)
    std = np.asarray(data.std)
    if mean.size != channels or std.size != channels:
        raise DatasetValidationError("mean/std must contain one value per channel")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise DatasetValidationError("mean/std must be finite and std must be positive")
    dates = np.asarray(data.dates).astype(str)
    if len(dates) != timesteps:
        raise DatasetValidationError("dates length does not match x")
    if len(dates) > 1 and np.any(dates[1:] <= dates[:-1]):
        raise DatasetValidationError("dates must be strictly increasing")
    names = np.asarray(data.names).astype(str)
    if len(names) != nodes or len(set(names)) != nodes:
        raise DatasetValidationError("names must contain one unique name per node")
    target_columns = np.asarray(data.target_columns).astype(str)
    if len(target_columns) != channels or len(set(target_columns)) != channels:
        raise DatasetValidationError("target_columns must contain one unique label per channel")
    if not 0 < int(data.train_end) < int(data.val_end) <= timesteps:
        raise DatasetValidationError("train_end and val_end must satisfy 0 < train_end < val_end <= T")
    return data


def load_graph_tmf(path: str | Path, *, validate: bool = True) -> GraphTMFData:
    source_path = Path(path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    with np.load(source_path, allow_pickle=True) as payload:
        missing = [key for key in ("x", "adj", "mean", "std", "dates", "names", "train_end", "val_end", "target") if key not in payload]
        if missing:
            raise DatasetValidationError(f"NPZ is missing required keys: {', '.join(missing)}")
        x = np.asarray(payload["x"], dtype=np.float32)
        channels = int(x.shape[-1]) if x.ndim == 3 else 0
        target = _scalar(payload["target"])
        if "target_columns" in payload:
            target_columns = np.asarray(payload["target_columns"]).astype(str)
        elif channels == 1:
            target_columns = np.asarray([target or "OT"])
        else:
            target_columns = np.asarray([f"target_{index}" for index in range(channels)])
        data = GraphTMFData(
            x=x,
            adj=np.asarray(payload["adj"], dtype=np.float32),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            dates=np.asarray(payload["dates"]).astype(str),
            names=np.asarray(payload["names"]).astype(str),
            target_columns=target_columns,
            target=target,
            train_end=int(np.asarray(payload["train_end"]).item()),
            val_end=int(np.asarray(payload["val_end"]).item()),
            source_path=source_path,
            fingerprint=sha256_file(source_path),
        )
    return validate_graph_tmf(data) if validate else data


def apply_input_missingness(x: torch.Tensor, missing_rate: float, mode: str, seed: int) -> torch.Tensor:
    """Apply the historical deterministic missing-history corruption exactly."""
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1)")
    if mode == "none" or missing_rate == 0:
        return x
    if mode == "random":
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        mask = torch.rand(x.shape, generator=generator, device=x.device) >= missing_rate
    elif mode == "block":
        block_len = max(1, int(round(x.shape[0] * missing_rate)))
        span = x.shape[0] - block_len + 1
        start = (seed * 1_103_515_245 + 12_345) % span
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[start : start + block_len] = False
    else:
        raise ValueError("missing_mode must be one of: none, random, block")
    return torch.where(mask, x, torch.zeros_like(x))


class WindowDataset(Dataset):
    """Chronological windows preserving the original split-boundary semantics."""

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        seq_len: int,
        pred_len: int,
        start: int,
        end: int,
        missing_rate: float = 0.0,
        missing_mode: str = "none",
        seed: int = 2026,
        time_features: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        if seq_len <= 0 or pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if start < 0 or end < start:
            raise ValueError("window start/end are invalid")
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.start = int(start)
        self.end = int(end)
        self.length = max(0, self.end - self.start - self.seq_len - self.pred_len + 1)
        self.missing_rate = float(missing_rate)
        self.missing_mode = str(missing_mode)
        self.seed = int(seed)
        if time_features is None:
            self.time_features = None
        else:
            validated = validate_time_features(
                np.asarray(time_features),
                n_timesteps=len(self.data),
            )
            self.time_features = torch.as_tensor(validated, dtype=torch.float32)
        if self.missing_mode not in {"none", "random", "block"}:
            raise ValueError("missing_mode must be one of: none, random, block")

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        pos = self.start + idx
        x = self.data[pos : pos + self.seq_len].clone()
        y = self.data[pos + self.seq_len : pos + self.seq_len + self.pred_len]
        if self.missing_rate > 0:
            x = apply_input_missingness(x, self.missing_rate, self.missing_mode, self.seed + pos)
        if self.time_features is not None:
            time = self.time_features[pos : pos + self.seq_len].clone()
            return x, y, time
        return x, y
