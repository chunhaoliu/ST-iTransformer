"""Build aligned graph forecasting tensors from restricted TMF CSV files."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from utils.paths import resolve_data_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = resolve_data_root()
DEFAULT_SOURCE_CSV_ROOT = DATA_ROOT / "processed" / "49_intersections_5min"
DEFAULT_METADATA = DATA_ROOT / "metadata" / "intersection_metadata.json"
DEFAULT_ADJ = DATA_ROOT / "metadata" / "adj_matrix_threshold_2km.npy"
DEFAULT_DIST = DATA_ROOT / "metadata" / "distance_matrix_km.npy"
DEFAULT_OUTPUT = DATA_ROOT / "processed" / "graph_tmf_15_ot.npz"
DEFAULT_STATS = PROJECT_ROOT / "results" / "diagnostics" / "tmf_graph_dataset_15_stats.csv"
DEFAULT_CENTER = "Main Street East and Mall Entrance"
TMF12_COLS = ["EBL", "EBR", "EBT", "NBL", "NBR", "NBT", "SBL", "SBR", "SBT", "WBL", "WBR", "WBT"]
DATE_SUFFIX_RE = re.compile(r"(?P<name>.+?)(?P<suffix>\d{5,6}_\d{6})$")


def canonical_intersection_name(name: str) -> str:
    normalized = name.replace("/", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def strip_date_suffix(filename: str) -> str:
    stem = Path(filename).stem
    match = DATE_SUFFIX_RE.match(stem)
    return match.group("name") if match else stem


def load_available_files(source_csv_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(Path(source_csv_root).glob("*.csv")):
        files[canonical_intersection_name(strip_date_suffix(path.name))] = path
    return files


def load_metadata(metadata_path: Path) -> list[dict[str, object]]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def select_intersections(metadata: list[dict[str, object]], available_files: dict[str, Path], distance_matrix: np.ndarray, center_name: str = DEFAULT_CENTER, n_intersections: int = 15, min_rows: int = 200_000) -> list[int]:
    names = [str(item["name"]) for item in metadata]
    if center_name not in names:
        raise ValueError(f"Center intersection not found in metadata: {center_name}")
    center_idx = names.index(center_name)
    candidates = []
    for idx, name in enumerate(names):
        path = available_files.get(canonical_intersection_name(name))
        if path is None:
            continue
        row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1
        if row_count < min_rows:
            continue
        candidates.append((float(distance_matrix[center_idx, idx]), idx))
    selected = [idx for _, idx in sorted(candidates)[:n_intersections]]
    if center_idx not in selected:
        selected = [center_idx] + selected[: n_intersections - 1]
    if len(selected) < n_intersections:
        raise ValueError(f"Only found {len(selected)} candidate intersections with at least {min_rows} rows")
    return selected[:n_intersections]


def resolve_target_columns(target: str) -> list[str]:
    if target.upper() == "TMF12":
        return TMF12_COLS
    return [part.strip() for part in target.split(",") if part.strip()]


def normalize_node_frame(df: pd.DataFrame, target_columns: list[str], node_column: str) -> pd.DataFrame:
    normalized = df[["date", *target_columns]].copy()
    for column in target_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", *target_columns])
    normalized = normalized.groupby("date", as_index=False)[target_columns].mean()
    rename_map = {column: f"{node_column}__{column}" for column in target_columns}
    return normalized.rename(columns=rename_map).sort_values("date").reset_index(drop=True)


def build_aligned_frame(files: list[Path], target_columns: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for idx, path in enumerate(files):
        frame = pd.read_csv(path, usecols=["date", *target_columns])
        frame = normalize_node_frame(frame, target_columns, f"node_{idx}")
        merged = frame if merged is None else merged.merge(frame, on="date", how="inner")
    if merged is None or merged.empty:
        raise ValueError("No aligned records found for selected intersections")
    return merged.sort_values("date").reset_index(drop=True)


def save_stats(output_path: Path, names: list[str], files: list[Path], adj: np.ndarray, frame: pd.DataFrame, target_columns: list[str], stats_path: Path) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "value"])
        writer.writerow(["output", str(output_path)])
        writer.writerow(["n_intersections", len(names)])
        writer.writerow(["n_timesteps", len(frame)])
        writer.writerow(["start_date", frame["date"].iloc[0]])
        writer.writerow(["end_date", frame["date"].iloc[-1]])
        writer.writerow(["adj_edges", int(adj.sum())])
        writer.writerow(["avg_degree", float(adj.sum() / len(names))])
        writer.writerow(["target_columns", " | ".join(target_columns)])
        writer.writerow(["selected_intersections", " | ".join(names)])
        writer.writerow(["source_files", " | ".join(path.name for path in files)])


def build_graph_dataset(source_csv_root: Path = DEFAULT_SOURCE_CSV_ROOT, metadata_path: Path = DEFAULT_METADATA, adj_path: Path = DEFAULT_ADJ, distance_path: Path = DEFAULT_DIST, output_path: Path = DEFAULT_OUTPUT, stats_path: Path = DEFAULT_STATS, center_name: str = DEFAULT_CENTER, n_intersections: int = 15, target: str = "OT", min_rows: int = 200_000) -> None:
    available = load_available_files(source_csv_root)
    metadata = load_metadata(metadata_path)
    adj_all = np.load(adj_path)
    dist_all = np.load(distance_path)
    selected = select_intersections(metadata, available, dist_all, center_name, n_intersections, min_rows)
    target_columns = resolve_target_columns(target)
    names = [str(metadata[idx]["name"]) for idx in selected]
    files = [available[canonical_intersection_name(name)] for name in names]
    frame = build_aligned_frame(files, target_columns)
    value_columns = [f"node_{idx}__{column}" for idx in range(len(names)) for column in target_columns]
    raw = frame[value_columns].to_numpy(dtype=np.float32).reshape(len(frame), len(names), len(target_columns))
    train_end = int(len(raw) * 0.7)
    val_end = int(len(raw) * 0.8)
    mean = raw[:train_end].mean(axis=(0, 1), keepdims=True)
    std = np.where(raw[:train_end].std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, raw[:train_end].std(axis=(0, 1), keepdims=True))
    normalized = (raw - mean) / std
    adj = adj_all[np.ix_(selected, selected)].astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, x=normalized.astype(np.float32), x_raw=raw.astype(np.float32), adj=adj, names=np.array(names), dates=frame["date"].to_numpy(), mean=mean.astype(np.float32), std=std.astype(np.float32), train_end=np.array(train_end), val_end=np.array(val_end), target=np.array(target), target_columns=np.array(target_columns))
    save_stats(output_path, names, files, adj, frame, target_columns, stats_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a graph TMF dataset from the canonical processed CSV root.")
    parser.add_argument("--source-csv-root", type=Path, default=DEFAULT_SOURCE_CSV_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJ)
    parser.add_argument("--distance", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--center", default=DEFAULT_CENTER)
    parser.add_argument("--n-intersections", type=int, default=15)
    parser.add_argument("--target", default="OT")
    parser.add_argument("--min-rows", type=int, default=200_000)
    args = parser.parse_args(argv)
    build_graph_dataset(args.source_csv_root, args.metadata, args.adjacency, args.distance, args.output, args.stats, args.center, args.n_intersections, args.target, args.min_rows)
    print(f"Graph dataset written to {args.output}")
    print(f"Stats written to {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
