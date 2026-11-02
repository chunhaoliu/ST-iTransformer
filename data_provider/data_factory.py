"""Factory for validated graph TMF datasets and chronological loaders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.reproducibility import seed_worker
from utils.paths import resolve_data_path

from .graph_tmf import GraphTMFData, WindowDataset, load_graph_tmf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SplitWindowInfo:
    name: str
    start: int
    end: int
    count: int
    first_target_timestamp: str | None
    last_target_timestamp: str | None


@dataclass(frozen=True)
class LoaderBundle:
    dataset: GraphTMFData
    train: WindowDataset
    validation: WindowDataset
    test: WindowDataset
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    split_info: tuple[SplitWindowInfo, ...]




def load_dataset(path: str | Path) -> GraphTMFData:
    return load_graph_tmf(resolve_data_path(path))


def _split_info(name: str, dataset: WindowDataset, dates: object) -> SplitWindowInfo:
    date_array = dates
    first = None
    last = None
    if len(dataset) > 0:
        first = str(date_array[dataset.start + dataset.seq_len])
        last = str(date_array[dataset.end - 1])
    return SplitWindowInfo(name, dataset.start, dataset.end, len(dataset), first, last)


def build_loaders(
    dataset: GraphTMFData,
    *,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    missing_rate: float = 0.0,
    missing_mode: str = "none",
    seed: int = 2026,
    num_workers: int = 0,
    time_features: np.ndarray | None = None,
) -> LoaderBundle:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    train = WindowDataset(
        dataset.x, seq_len, pred_len, 0, dataset.train_end,
        missing_rate, missing_mode, seed, time_features=time_features,
    )
    validation = WindowDataset(
        dataset.x, seq_len, pred_len, max(0, dataset.train_end - seq_len), dataset.val_end,
        missing_rate, missing_mode, seed + 10_000, time_features=time_features,
    )
    test = WindowDataset(
        dataset.x, seq_len, pred_len, max(0, dataset.val_end - seq_len), len(dataset.x),
        missing_rate, missing_mode, seed + 20_000, time_features=time_features,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": num_workers, "worker_init_fn": seed_worker}
    train_loader = DataLoader(train, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation, shuffle=False, **common)
    test_loader = DataLoader(test, shuffle=False, **common)
    info = (
        _split_info("train", train, dataset.dates),
        _split_info("validation", validation, dataset.dates),
        _split_info("test", test, dataset.dates),
    )
    return LoaderBundle(dataset, train, validation, test, train_loader, validation_loader, test_loader, info)
