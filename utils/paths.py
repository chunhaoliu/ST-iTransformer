"""Portable runtime paths for the separated TMF code and material trees."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_ENV = "TMF_DATA_ROOT"
MANUSCRIPT_ROOT_ENV = "TMF_MANUSCRIPT_ROOT"


def _configured_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path when set: {raw!r}")
    return path.resolve()


def resolve_data_root(project_root: Path = PROJECT_ROOT) -> Path:
    """Return the external data root or the ignored repository-local fallback."""

    return _configured_path(DATA_ROOT_ENV) or (Path(project_root).resolve() / "data")


def resolve_manuscript_root() -> Path | None:
    """Return the manuscript root only when the caller explicitly configures it."""

    return _configured_path(MANUSCRIPT_ROOT_ENV)


def _data_relative_path(path: Path) -> Path:
    """Normalize ``data/...`` and legacy ``../data/...`` spellings."""

    parts = list(path.parts)
    while parts and parts[0] in {".", ".."}:
        parts.pop(0)
    if parts and parts[0].lower() == "data":
        parts.pop(0)
    return Path(*parts) if parts else Path(".")


def resolve_data_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a dataset path across the separated code/material boundary."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    root = Path(project_root).resolve()
    relative_data = _data_relative_path(path)
    configured_root = _configured_path(DATA_ROOT_ENV)
    candidates: list[Path] = []
    if configured_root is not None:
        candidates.append(configured_root / relative_data)
    candidates.extend([root / relative_data, root / path, root.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()
