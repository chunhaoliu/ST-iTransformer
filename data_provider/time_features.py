"""Deterministic calendar covariates derived from observed timestamps."""
from __future__ import annotations

from datetime import datetime

import numpy as np


TIME_FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
)
TIME_FEATURE_DIM = len(TIME_FEATURE_NAMES)


def _parse_timestamp(value: object) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


def build_calendar_features(dates: np.ndarray | list[object]) -> np.ndarray:
    """Return bounded cyclical hour, weekday, and month features for each date.

    The returned array has shape [T, 6]. Only timestamp metadata is used;
    no flow values or target timestamps are accessed.
    """

    values = np.asarray(dates).reshape(-1)
    features = np.empty((len(values), TIME_FEATURE_DIM), dtype=np.float32)
    for index, value in enumerate(values):
        timestamp = _parse_timestamp(value)
        hour_fraction = (
            timestamp.hour
            + timestamp.minute / 60.0
            + timestamp.second / 3600.0
        ) / 24.0
        weekday_fraction = timestamp.weekday() / 7.0
        month_fraction = (timestamp.month - 1) / 12.0
        features[index] = (
            np.sin(2.0 * np.pi * hour_fraction),
            np.cos(2.0 * np.pi * hour_fraction),
            np.sin(2.0 * np.pi * weekday_fraction),
            np.cos(2.0 * np.pi * weekday_fraction),
            np.sin(2.0 * np.pi * month_fraction),
            np.cos(2.0 * np.pi * month_fraction),
        )
    return features


def validate_time_features(
    time_features: np.ndarray,
    *,
    n_timesteps: int,
    expected_dim: int = TIME_FEATURE_DIM,
) -> np.ndarray:
    """Validate a full-timeline feature array before window slicing."""

    values = np.asarray(time_features, dtype=np.float32)
    if values.shape != (n_timesteps, expected_dim):
        raise ValueError(
            f"time_features must have shape {(n_timesteps, expected_dim)}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("time_features contain non-finite values")
    if np.abs(values).max(initial=0.0) > 1.000001:
        raise ValueError("cyclical time_features must be bounded by [-1, 1]")
    return values
