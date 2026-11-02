"""Load and validate the frozen TMF protocol and experiment recipes."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "protocols" / "tmf_realtime_short_suite.yaml"
SCHEMA_VERSION = 1
SHORT_SUITE_PROTOCOL_ID = "tmf_st_itransformer_realtime_short_suite_v1"
SUPPORTED_PROTOCOL_IDS = {SHORT_SUITE_PROTOCOL_ID}
EXPECTED_LOOKBACKS = {
    SHORT_SUITE_PROTOCOL_ID: [1, 2, 6, 12, 24],
}

EXPECTED_TMF12_HORIZONS = {
    SHORT_SUITE_PROTOCOL_ID: [1, 2, 3, 4, 6, 8, 12, 16, 20, 24],
}


class ProtocolValidationError(ValueError):
    """Raised when a versioned protocol or recipe violates its contract."""


def is_absolute_versioned_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return Path(value).is_absolute() or bool(PureWindowsPath(value).drive)


def resolve_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def portable_path(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ProtocolValidationError(f"{context} is missing required keys: {', '.join(missing)}")


def _require_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProtocolValidationError(f"{name} must be a positive integer")
    return int(value)


def _require_positive_int_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ProtocolValidationError(f"{name} must be a non-empty list")
    result = [_require_positive_int(item, f"{name} item") for item in value]
    if len(set(result)) != len(result):
        raise ProtocolValidationError(f"{name} must not contain duplicates")
    return result


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Validate the canonical protocol without mutating scientific values."""
    if not isinstance(protocol, dict):
        raise ProtocolValidationError("protocol root must be a mapping")
    _require(
        protocol,
        ("schema_version", "protocol_id", "pipeline_id", "roots", "task", "datasets", "data_protocol", "models", "training", "metrics", "artifact_sources"),
        "protocol",
    )
    if protocol["schema_version"] != SCHEMA_VERSION:
        raise ProtocolValidationError(
            f"unsupported schema_version={protocol['schema_version']!r}; expected {SCHEMA_VERSION}"
        )
    protocol_id = str(protocol["protocol_id"])
    if protocol_id not in SUPPORTED_PROTOCOL_IDS:
        raise ProtocolValidationError(
            f"unsupported protocol_id: {protocol['protocol_id']!r}; expected one of "
            f"{sorted(SUPPORTED_PROTOCOL_IDS)!r}"
        )
    if not isinstance(protocol["pipeline_id"], str) or not protocol["pipeline_id"].strip():
        raise ProtocolValidationError("pipeline_id must be a non-empty string")

    roots = protocol["roots"]
    if not isinstance(roots, dict):
        raise ProtocolValidationError("roots must be a mapping")
    _require(roots, ("workspace", "code", "manuscript", "restricted_data"), "roots")
    for name, value in roots.items():
        if is_absolute_versioned_path(value):
            raise ProtocolValidationError(f"roots.{name} must be relative, got {value!r}")

    task = protocol["task"]
    if not isinstance(task, dict):
        raise ProtocolValidationError("task must be a mapping")
    _require(task, ("sampling_interval", "input_shape", "output_shape", "lookback_steps", "targets"), "task")
    if task["input_shape"] != ["batch", "lookback", "nodes", "channels"]:
        raise ProtocolValidationError("task.input_shape does not match the [B,L,N,C] contract")
    if task["output_shape"] != ["batch", "horizon", "nodes", "channels"]:
        raise ProtocolValidationError("task.output_shape does not match the [B,H,N,C] contract")
    actual_lookbacks = (
        _require_positive_int_list(task["lookback_steps"], "task.lookback_steps")
        if isinstance(task["lookback_steps"], list)
        else [_require_positive_int(task["lookback_steps"], "task.lookback_steps")]
    )
    expected_lookbacks = EXPECTED_LOOKBACKS[protocol_id]
    if actual_lookbacks != expected_lookbacks:
        raise ProtocolValidationError(
            f"task.lookback_steps must remain {expected_lookbacks} for {protocol_id}"
        )

    datasets = protocol["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ProtocolValidationError("datasets must be a non-empty list")
    dataset_ids: set[str] = set()
    for index, dataset in enumerate(datasets):
        context = f"datasets[{index}]"
        if not isinstance(dataset, dict):
            raise ProtocolValidationError(f"{context} must be a mapping")
        _require(dataset, ("id", "path", "nodes", "channels", "role"), context)
        if dataset["id"] in dataset_ids:
            raise ProtocolValidationError(f"duplicate dataset id: {dataset['id']}")
        dataset_ids.add(str(dataset["id"]))
        if is_absolute_versioned_path(dataset["path"]):
            raise ProtocolValidationError(f"{context}.path must be relative")
        _require_positive_int(dataset["nodes"], f"{context}.nodes")
        _require_positive_int(dataset["channels"], f"{context}.channels")

    data_protocol = protocol["data_protocol"]
    if not isinstance(data_protocol, dict):
        raise ProtocolValidationError("data_protocol must be a mapping")
    _require(data_protocol, ("split", "dataset_normalization", "instance_normalization", "adjacency", "restricted_data"), "data_protocol")
    if "70" not in str(data_protocol["split"]) or "10" not in str(data_protocol["split"]) or "20" not in str(data_protocol["split"]):
        raise ProtocolValidationError("data_protocol.split must preserve the chronological 70/10/20 split")

    models = protocol["models"]
    if not isinstance(models, dict):
        raise ProtocolValidationError("models must be a mapping")
    _require(models, ("canonical_model_class", "matched_baseline_class", "legacy_model_class"), "models")
    if models["canonical_model_class"] != "models.i_transformer.GraphITransformerForecaster":
        raise ProtocolValidationError("canonical model class changed")

    training = protocol["training"]
    if not isinstance(training, dict):
        raise ProtocolValidationError("training must be a mapping")
    seeds = _require_positive_int_list(training.get("seeds"), "training.seeds")
    if seeds != [2026, 2027, 2028]:
        raise ProtocolValidationError("training.seeds must remain [2026, 2027, 2028]")
    if protocol_id != SHORT_SUITE_PROTOCOL_ID:
        if _require_positive_int_list(training.get("horizons_ot"), "training.horizons_ot") != [96, 192, 336, 720]:
            raise ProtocolValidationError("training.horizons_ot changed")
    expected_tmf12_horizons = EXPECTED_TMF12_HORIZONS[protocol_id]
    if _require_positive_int_list(training.get("horizons_tmf12"), "training.horizons_tmf12") != expected_tmf12_horizons:
        raise ProtocolValidationError(
            f"training.horizons_tmf12 must remain {expected_tmf12_horizons} for {protocol_id}"
        )
    for name in ("d_model", "n_heads", "encoder_layers", "feed_forward_width", "epochs", "batch_size_ot", "batch_size_tmf12"):
        _require_positive_int(training.get(name), f"training.{name}")
    if float(training.get("learning_rate", 0)) <= 0:
        raise ProtocolValidationError("training.learning_rate must be positive")

    metrics = protocol["metrics"]
    if not isinstance(metrics, dict):
        raise ProtocolValidationError("metrics must be a mapping")
    _require(metrics, ("primary", "aggregation", "raw_scale_diagnostics"), "metrics")

    artifact_sources = protocol["artifact_sources"]
    if not isinstance(artifact_sources, dict):
        raise ProtocolValidationError("artifact_sources must be a mapping")
    for name, value in artifact_sources.items():
        if isinstance(value, str) and is_absolute_versioned_path(value):
            raise ProtocolValidationError(f"artifact_sources.{name} must be relative")
    return protocol


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    return validate_protocol(protocol)


def validate_recipe(recipe: dict[str, Any], protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a declarative recipe and reject embedded commands/absolute paths."""
    if not isinstance(recipe, dict):
        raise ProtocolValidationError("recipe root must be a mapping")
    _require(recipe, ("schema_version", "recipe_id", "protocol_id", "data", "models", "training", "evaluation", "output"), "recipe")
    if recipe["schema_version"] != SCHEMA_VERSION:
        raise ProtocolValidationError("recipe schema_version is unsupported")
    if protocol is not None and recipe["protocol_id"] != protocol["protocol_id"]:
        raise ProtocolValidationError("recipe protocol_id does not match the canonical protocol")
    if "protocol_path" in recipe and is_absolute_versioned_path(recipe["protocol_path"]):
        raise ProtocolValidationError("recipe.protocol_path must be relative")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(recipe["recipe_id"])):
        raise ProtocolValidationError("recipe_id contains unsupported characters")
    for section_name in ("data", "training", "evaluation", "output"):
        if not isinstance(recipe[section_name], dict):
            raise ProtocolValidationError(f"recipe.{section_name} must be a mapping")
    if "prediction" in recipe:
        prediction = recipe["prediction"]
        if not isinstance(prediction, dict):
            raise ProtocolValidationError("recipe.prediction must be a mapping")
        if "output_template" not in prediction:
            raise ProtocolValidationError("recipe.prediction requires output_template")
        if is_absolute_versioned_path(prediction["output_template"]):
            raise ProtocolValidationError("recipe.prediction.output_template must use a relative path")
        for key in ("node_index", "start", "count"):
            if key in prediction and (not isinstance(prediction[key], int) or isinstance(prediction[key], bool) or prediction[key] < 0):
                raise ProtocolValidationError(f"recipe.prediction.{key} must be a non-negative integer")
    data = recipe["data"]
    _require(data, ("target",), "recipe.data")
    if "dataset" not in data and "datasets" not in data:
        raise ProtocolValidationError("recipe.data requires dataset or datasets")
    datasets = data.get("datasets", [data])
    if not isinstance(datasets, list) or not datasets:
        raise ProtocolValidationError("recipe.data.datasets must be a non-empty list")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ProtocolValidationError(f"recipe.data.datasets[{index}] must be a mapping")
        _require(dataset, ("benchmark", "dataset"), f"recipe.data.datasets[{index}]")
        if is_absolute_versioned_path(dataset["dataset"]):
            raise ProtocolValidationError(f"recipe.data.datasets[{index}].dataset must be relative")
    models = recipe["models"]
    if not isinstance(models, list) or not models:
        raise ProtocolValidationError("recipe.models must be a non-empty list")
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ProtocolValidationError(f"recipe.models[{index}] must be a mapping")
        _require(model, ("id", "display_name"), f"recipe.models[{index}]")
        if "command" in model or "command" in recipe:
            raise ProtocolValidationError("recipes may not contain executable command strings")
        if "context_blend_weight" in model:
            try:
                context_weight = float(model["context_blend_weight"])
            except (TypeError, ValueError) as exc:
                raise ProtocolValidationError(f"recipe.models[{index}].context_blend_weight must be numeric") from exc
            if not 0.0 <= context_weight <= 1.0:
                raise ProtocolValidationError(f"recipe.models[{index}].context_blend_weight must be in [0, 1]")
        if "context_mean_steps" in model:
            _require_positive_int(model["context_mean_steps"], f"recipe.models[{index}].context_mean_steps")
        if "context_gate_enabled" in model and not isinstance(model["context_gate_enabled"], bool):
            raise ProtocolValidationError(f"recipe.models[{index}].context_gate_enabled must be boolean")
        if "context_gate_bias" in model:
            try:
                float(model["context_gate_bias"])
            except (TypeError, ValueError) as exc:
                raise ProtocolValidationError(f"recipe.models[{index}].context_gate_bias must be numeric") from exc
        if "input_context_mode" in model and str(model["input_context_mode"]) not in {"none", "raw"}:
            raise ProtocolValidationError(
                f"recipe.models[{index}].input_context_mode is invalid"
            )
        if "spatial_mask_mode" in model and str(model["spatial_mask_mode"]) not in {
            "proximity", "dense", "permuted", "soft_proximity"
        }:
            raise ProtocolValidationError(
                f"recipe.models[{index}].spatial_mask_mode is invalid"
            )
        if "spatial_mask_strength" in model:
            try:
                spatial_mask_strength = float(model["spatial_mask_strength"])
            except (TypeError, ValueError) as exc:
                raise ProtocolValidationError(
                    f"recipe.models[{index}].spatial_mask_strength must be numeric"
                ) from exc
            if spatial_mask_strength < 0:
                raise ProtocolValidationError(
                    f"recipe.models[{index}].spatial_mask_strength must be non-negative"
                )
        if "loss_name" in model:
            loss_name = str(model["loss_name"]).strip().lower()
            if loss_name not in {"mse", "smooth_l1"}:
                raise ProtocolValidationError(
                    f"recipe.models[{index}].loss_name must be one of mse or smooth_l1"
                )
        if "weight_decay" in model:
            try:
                weight_decay = float(model["weight_decay"])
            except (TypeError, ValueError) as exc:
                raise ProtocolValidationError(f"recipe.models[{index}].weight_decay must be numeric") from exc
            if weight_decay < 0:
                raise ProtocolValidationError(f"recipe.models[{index}].weight_decay must be non-negative")
    training = recipe["training"]
    _require(training, ("seq_len", "pred_len", "seeds"), "recipe.training")
    if isinstance(training["seq_len"], list):
        seq_lengths = _require_positive_int_list(training["seq_len"], "recipe.training.seq_len")
    else:
        seq_lengths = [_require_positive_int(training["seq_len"], "recipe.training.seq_len")]
    if protocol is not None:
        protocol_lookbacks = protocol["task"]["lookback_steps"]
        allowed_seq_lengths = (
            protocol_lookbacks
            if isinstance(protocol_lookbacks, list)
            else [protocol_lookbacks]
        )
        if not set(seq_lengths).issubset(set(allowed_seq_lengths)):
            raise ProtocolValidationError(
                "recipe.training.seq_len must be contained in protocol.task.lookback_steps; "
                f"got {seq_lengths!r}, allowed {allowed_seq_lengths!r}"
            )
    pred_lengths = _require_positive_int_list(training["pred_len"], "recipe.training.pred_len")
    if protocol is not None and str(data.get("target", "")).upper() == "TMF12":
        allowed_pred_lengths = protocol["training"]["horizons_tmf12"]
        if not set(pred_lengths).issubset(set(allowed_pred_lengths)):
            raise ProtocolValidationError(
                "recipe.training.pred_len must be contained in protocol.training.horizons_tmf12; "
                f"got {pred_lengths!r}, allowed {allowed_pred_lengths!r}"
            )
    _require_positive_int_list(training["seeds"], "recipe.training.seeds")
    evaluation = recipe["evaluation"]
    if "missing_mode" in evaluation and evaluation["missing_mode"] not in {"none", "random", "block"}:
        raise ProtocolValidationError("recipe.evaluation.missing_mode is invalid")
    for section_name, section in recipe.items():
        if isinstance(section, dict):
            for key, value in section.items():
                if key.endswith("path") or key in {"dataset", "metrics_dir", "diagnostics_dir", "history_dir"}:
                    values = value if isinstance(value, list) else [value]
                    if any(is_absolute_versioned_path(item) for item in values if isinstance(item, str)):
                        raise ProtocolValidationError(f"recipe.{section_name}.{key} must use relative paths")
    return recipe


def json_fingerprint(value: object) -> str:
    """Stable short fingerprint for validated metadata mappings."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
