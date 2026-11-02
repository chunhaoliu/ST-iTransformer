"""Atomic, provenance-aware serialization for experiment results."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RESULT_FIELDNAMES = [
    "benchmark", "target", "model", "n_nodes", "seq_len", "pred_len", "epochs", "batch_size",
    "d_model", "n_heads", "e_layers", "use_spatial", "use_fusion", "spatial_source", "physics_weight",
    "missing_mode", "input_missing_rate", "seed", "max_train_batches", "max_eval_batches", "test_mse",
    "test_mae", "test_physics_error", "n_parameters", "train_seconds", "eval_seconds", "total_seconds",
]
RESULT_FIELDNAMES = LEGACY_RESULT_FIELDNAMES + [
    "schema_version", "protocol_id", "recipe_id", "run_id", "model_id", "model_display_name",
    "spatial_mask_mode", "spatial_mask_seed", "spatial_mask_strength",
    "spatial_layers", "spatial_residual_scale", "fusion_gate_bias", "fusion_mode",
    "context_blend_weight", "context_mean_steps",
    "context_gate_enabled", "context_gate_bias", "input_context_mode", "loss_name",

    "d_ff", "dropout", "lr", "weight_decay", "patience",
    "node_count", "channel_count", "adjacency_mode", "data_path", "data_fingerprint", "code_commit",
    "use_time_features",
    "status", "diagnostic_only", "condition_id",
]
RESULT_SCHEMA_VERSION = "tmf_result_v1"
IDENTITY_FIELDS = [
    "protocol_id", "recipe_id", "benchmark", "target", "model_id", "model_display_name",
    "spatial_source", "spatial_mask_mode", "spatial_mask_seed",
    "spatial_layers", "spatial_residual_scale", "fusion_gate_bias", "fusion_mode",
    "context_blend_weight", "context_mean_steps",
    "context_gate_enabled", "context_gate_bias", "input_context_mode", "loss_name",

    "epochs", "batch_size", "d_model", "n_heads", "e_layers", "d_ff", "dropout", "lr", "weight_decay", "patience",
    "max_train_batches", "max_eval_batches", "seq_len",
    "pred_len", "seed", "node_count", "channel_count", "adjacency_mode", "missing_mode",
    "input_missing_rate", "physics_weight", "condition_id", "code_commit", "data_fingerprint",
]


class DuplicateRunError(ValueError):
    """Raised when a result identity already exists without explicit overwrite."""


def portable_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def stable_run_id(fields: dict[str, object]) -> str:
    payload = {key: fields.get(key, "") for key in IDENTITY_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"tmf-{hashlib.sha256(encoded).hexdigest()[:16]}"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_records(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    """Atomically write a non-run CSV such as history or diagnostics."""
    _atomic_csv(Path(path), fieldnames, list(rows))


def _normalized_row(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized.setdefault("schema_version", RESULT_SCHEMA_VERSION)
    normalized.setdefault("status", "completed")
    normalized.setdefault("diagnostic_only", False)
    normalized.setdefault("use_time_features", False)
    normalized.setdefault("input_context_mode", "none")
    normalized.setdefault("run_id", stable_run_id(normalized))
    if "data_path" in normalized and normalized["data_path"]:
        normalized["data_path"] = portable_path(str(normalized["data_path"]))
    return normalized


def _legacy_identity(row: dict[str, object]) -> tuple[str, ...]:
    """Identity fallback for historical CSVs that predate ``run_id``."""
    fields = (
        "benchmark", "target", "model", "n_nodes", "seq_len", "pred_len", "seed",
        "physics_weight", "missing_mode", "input_missing_rate",
    )
    return tuple(str(row.get(field, "")) for field in fields)


def write_result(
    output_path: str | Path,
    row: dict[str, object],
    *,
    overwrite_run: bool = False,
    legacy: bool = False,
) -> dict[str, object]:
    """Write one result atomically while preserving legacy CSV columns."""
    path = Path(output_path)
    normalized = _normalized_row(row)
    if legacy:
        fields = LEGACY_RESULT_FIELDNAMES
        existing_fields, existing_rows = read_rows(path) if path.exists() else ([], [])
        if existing_fields and existing_fields != fields:
            fields = existing_fields
        _atomic_csv(path, fields, [*existing_rows, normalized])
        return normalized

    existing_fields, existing_rows = read_rows(path) if path.exists() else ([], [])
    fields = existing_fields or RESULT_FIELDNAMES
    existing_ids = {
        row.get("run_id") or stable_run_id({key: row.get(key, "") for key in IDENTITY_FIELDS})
        for row in existing_rows
    }
    run_id = str(normalized["run_id"])
    legacy_key = _legacy_identity(normalized)
    legacy_duplicate = any(_legacy_identity(existing) == legacy_key for existing in existing_rows)
    if (run_id in existing_ids or (fields != RESULT_FIELDNAMES and legacy_duplicate)) and not overwrite_run:
        detail = run_id if fields == RESULT_FIELDNAMES else legacy_key
        raise DuplicateRunError(f"result identity already exists in {path}: {detail}")
    if run_id in existing_ids or (fields != RESULT_FIELDNAMES and legacy_duplicate):
        replaced = []
        for existing in existing_rows:
            existing_id = existing.get("run_id") or stable_run_id({key: existing.get(key, "") for key in IDENTITY_FIELDS})
            same_legacy = fields != RESULT_FIELDNAMES and _legacy_identity(existing) == legacy_key
            replaced.append(normalized if existing_id == run_id or same_legacy else existing)
        existing_rows = replaced
    else:
        existing_rows.append(normalized)
    _atomic_csv(path, fields, existing_rows)
    return normalized


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
