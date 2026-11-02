"""Multi-horizon TMF benchmark (v2): shared protocol, validation selection, raw metrics.

Protocol (configs/experiments/benchmark_v2.json): 60-minute history and direct
60-minute forecasts (L = H = 12 five-minute steps), chronological 70/10/20
splits with training-only scaling, mean absolute error on vehicle counts as
the shared training loss, validation MAE for learning-rate selection and
checkpoint selection, and non-negative (clipped) vehicle-count forecasts for
every model at evaluation.  The test split is evaluated only by final runs,
after the model's learning rate has been locked by validation.

Execution: ``prepare`` copies the reviewed source files into a fresh
``source/`` snapshot under the output root and records their hashes;
``pipeline`` starts the requested number of workers per GPU from that
snapshot.  Workers claim jobs with atomic lock directories, so a job runs
once; a completed run is never rerun or overwritten.  An interrupted run
folder is moved to ``_interrupted/`` and restarted from scratch.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/experiments/benchmark_v2.json"
SNAPSHOT_FILES = (
    "configs/experiments/benchmark_v2.json",
    "experiments/__init__.py",
    "experiments/benchmark_v2.py",
    "experiments/movement_graph.py",
)
# Whole packages, so that package-level imports resolve inside the snapshot.
SNAPSHOT_TREES = ("data_provider", "utils", "models", "exp")


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def plan_file(args):
    """Resolve a source-root-relative plan, also inside a prepared snapshot."""
    path = Path(getattr(args, "plan", PLAN))
    path = (path if path.is_absolute() else ROOT / path).resolve()
    path.relative_to(ROOT.resolve())  # plans must be snapshot-local
    return path


def snapshot_files(path):
    """Keep default snapshot names; substitute only the selected plan."""
    name = path.relative_to(ROOT.resolve()).as_posix()
    files = [name if item == "configs/experiments/benchmark_v2.json" else item
             for item in SNAPSHOT_FILES]
    for tree in SNAPSHOT_TREES:
        files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / tree).rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    if "himnet" in read_json(path)["models"]:
        from models.himnet_adapter import HASHES, load_official_class
        load_official_class()  # refuse drift before preparing a snapshot
        files += [f"external/himnet/{name}" for name in HASHES]
    return files


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def calendar_indices(dates) -> np.ndarray:
    """[T, 3] int64: five-minute slot, weekday, month index of each source label."""
    out = np.empty((len(dates), 3), dtype=np.int64)
    for i, value in enumerate(dates):
        stamp = datetime.fromisoformat(str(value))
        out[i] = (stamp.hour * 12 + stamp.minute // 5, stamp.weekday(), stamp.month - 1)
    return out


class Windows:
    """Chronological (x, y, cal) windows; targets stay inside their split."""

    def __init__(self, x, cal, seq_len, pred_len, start, end, future_calendar=False):
        import torch
        self.future_calendar = future_calendar
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.cal = torch.as_tensor(cal, dtype=torch.long)
        self.seq_len, self.pred_len, self.start, self.end = seq_len, pred_len, start, end
        self.length = max(0, end - start - seq_len - pred_len + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        position = self.start + index
        history = slice(position, position + self.seq_len)
        future = slice(position + self.seq_len, position + self.seq_len + self.pred_len)
        calendar = slice(position, position + self.seq_len + self.pred_len) if self.future_calendar else history
        return self.x[history], self.x[future], self.cal[calendar]

    def target_positions(self):
        """Absolute timeline index of every (window, step) target."""
        base = self.start + self.seq_len + np.arange(self.length)
        return base[:, None] + np.arange(self.pred_len)[None, :]


def split_windows(data, cal, seq_len, pred_len, future_calendar=False):
    train = Windows(data.x, cal, seq_len, pred_len, 0, data.train_end, future_calendar)
    validation = Windows(data.x, cal, seq_len, pred_len, data.train_end - seq_len, data.val_end, future_calendar)
    test = Windows(data.x, cal, seq_len, pred_len, data.val_end - seq_len, len(data.x), future_calendar)
    return train, validation, test


def demand_thresholds(data):
    """Per-series 95th percentile of positive training counts (no val/test labels)."""
    raw = data.x[: data.train_end].astype(np.float64) * data.std.reshape(1, 1, -1) + data.mean.reshape(1, 1, -1)
    thresholds = np.full(raw.shape[1:], np.inf)
    for n in range(raw.shape[1]):
        for c in range(raw.shape[2]):
            positive = raw[:, n, c][raw[:, n, c] > 1e-3]
            if positive.size:
                thresholds[n, c] = np.quantile(positive, 0.95)
    return thresholds


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------

def make_model(plan, name, options, data, token_graphs, official_source):
    from models import tmf_v2
    graph_key = options.get("graph", "movement")
    return tmf_v2.build(name, seq_len=options.get("seq_len", plan["seq_len"]), pred_len=plan["pred_len"],
                        n_nodes=data.n_nodes, n_channels=data.n_channels,
                        token_graph=token_graphs[graph_key], official_source=official_source,
                        **{k: v for k, v in options.items() if k not in ("graph", "seq_len")})


def raw_scale(tensor, mean, std):
    return tensor * std + mean


def run_epoch(model, loader, device, mean, std, optimizer=None, max_batches=None, clip=5.0):
    import torch
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for index, (x, y, cal) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            x, y, cal = x.to(device, non_blocking=True), y.to(device, non_blocking=True), cal.to(device, non_blocking=True)
            hook = getattr(model, "training_forward", None) if training else None
            prediction = hook(x, cal, y) if hook is not None else model(x, cal)
            error = (raw_scale(prediction, mean, std) - raw_scale(y, mean, std)).abs()
            loss = error.mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total += float(error.detach().sum())
            count += error.numel()
    return total / count


def evaluate_test(model, loader, windows, data, device, thresholds):
    """Streaming raw-count metrics, demand/month partitions and per-window errors."""
    import torch
    model.eval()
    mean = torch.as_tensor(data.mean, device=device).reshape(1, 1, 1, -1)
    std = torch.as_tensor(data.std, device=device).reshape(1, 1, 1, -1)
    high = torch.as_tensor(thresholds, device=device, dtype=torch.float32)
    horizon = windows.pred_len
    months = np.array([str(d)[:7] for d in data.dates])
    positions = windows.target_positions()
    month_labels = sorted(set(months[positions.reshape(-1)]))
    month_index = {m: i for i, m in enumerate(month_labels)}
    target_month = np.vectorize(month_index.get)(months[positions])  # [W, H]
    n_groups, n_months = 3, len(month_labels)
    acc = {k: np.zeros((horizon,)) for k in ("abs", "abs_unclipped", "sq", "y", "n")}
    group = {k: np.zeros((n_groups, horizon)) for k in ("abs", "sq", "under", "bias", "n")}
    month = {k: np.zeros((n_months, horizon)) for k in ("abs", "sq", "n")}
    node_abs = np.zeros((data.n_nodes, data.n_channels))
    window_abs, window_sq = [], []
    offset = 0
    with torch.no_grad():
        for x, y, cal in loader:
            x, y, cal = x.to(device), y.to(device), cal.to(device)
            raw_pred = raw_scale(model(x, cal), mean, std)
            raw_true = raw_scale(y, mean, std)
            clipped = raw_pred.clamp_min(0.0)
            error = clipped - raw_true
            absolute = error.abs()
            batch = x.shape[0]
            acc["abs"] += absolute.sum(dim=(0, 2, 3)).double().cpu().numpy()
            acc["abs_unclipped"] += (raw_pred - raw_true).abs().sum(dim=(0, 2, 3)).double().cpu().numpy()
            acc["sq"] += (error ** 2).sum(dim=(0, 2, 3)).double().cpu().numpy()
            acc["y"] += raw_true.sum(dim=(0, 2, 3)).double().cpu().numpy()
            acc["n"] += batch * data.n_nodes * data.n_channels
            zero = raw_true.abs() <= 1e-3
            peak = (~zero) & (raw_true >= high)
            masks = (zero, (~zero) & (~peak), peak)
            under = (raw_true - clipped).clamp_min(0.0)
            for g, mask in enumerate(masks):
                m = mask.double()
                group["abs"][g] += (absolute * m).sum(dim=(0, 2, 3)).cpu().numpy()
                group["sq"][g] += (error ** 2 * m).sum(dim=(0, 2, 3)).cpu().numpy()
                group["under"][g] += (under * m).sum(dim=(0, 2, 3)).cpu().numpy()
                group["bias"][g] += (error * m).sum(dim=(0, 2, 3)).cpu().numpy()
                group["n"][g] += m.sum(dim=(0, 2, 3)).cpu().numpy()
            per_window_abs = absolute.mean(dim=(2, 3)).cpu().numpy()  # [B, H]
            per_window_sq = (error ** 2).mean(dim=(2, 3)).cpu().numpy()
            window_abs.append(per_window_abs.astype(np.float32))
            window_sq.append(per_window_sq.astype(np.float32))
            labels = target_month[offset: offset + batch]
            for mi in range(n_months):
                hit = labels == mi
                month["abs"][mi] += (per_window_abs * hit).sum(axis=0) * data.n_nodes * data.n_channels
                month["sq"][mi] += (per_window_sq * hit).sum(axis=0) * data.n_nodes * data.n_channels
                month["n"][mi] += hit.sum(axis=0) * data.n_nodes * data.n_channels
            node_abs += absolute.sum(dim=(0, 1)).double().cpu().numpy()
            offset += batch
    if offset != len(windows):
        raise ValueError("Incomplete test coverage")

    def summary(abs_, sq, n, y=None):
        out = dict(mae=(abs_ / n).tolist(), rmse=np.sqrt(sq / n).tolist(),
                   mae_all=float(abs_.sum() / n.sum()), rmse_all=float(np.sqrt(sq.sum() / n.sum())))
        if y is not None:
            out.update(wmape=(abs_ / y).tolist(), wmape_all=float(abs_.sum() / y.sum()))
        return out

    metrics = summary(acc["abs"], acc["sq"], acc["n"], acc["y"])
    metrics["mae_unclipped_all"] = float(acc["abs_unclipped"].sum() / acc["n"].sum())
    metrics["targets"] = int(acc["n"].sum())
    metrics["windows"] = int(offset)
    names = ("zero", "ordinary_positive", "high_positive")
    metrics["demand"] = {names[g]: dict(
        summary(group["abs"][g], group["sq"][g], group["n"][g]),
        underprediction_all=float(group["under"][g].sum() / group["n"][g].sum()),
        bias_all=float(group["bias"][g].sum() / group["n"][g].sum()),
        count=int(group["n"][g].sum())) for g in range(n_groups)}
    metrics["months"] = {label: dict(summary(month["abs"][i], month["sq"][i], month["n"][i]),
                                     count=int(month["n"][i].sum())) for i, label in enumerate(month_labels)}
    metrics["node_movement_mae"] = (node_abs / (offset * horizon)).tolist()
    return metrics, np.concatenate(window_abs), np.concatenate(window_sq)


def fit(plan, job, data, cal, token_graphs, official_source, device, folder, log):
    import torch
    from torch.utils.data import DataLoader
    from utils.reproducibility import seed_everything, seed_worker
    seed_everything(job["seed"])
    train, validation, test = split_windows(data, cal, job["options"].get("seq_len", plan["seq_len"]), plan["pred_len"], future_calendar=job["model"] == "himnet")
    generator = torch.Generator().manual_seed(job["seed"])
    common = dict(batch_size=plan["batch_size"], num_workers=plan["num_workers"],
                  worker_init_fn=seed_worker, pin_memory=True)
    train_loader = DataLoader(train, shuffle=True, generator=generator, drop_last=False, **common)
    validation_loader = DataLoader(validation, shuffle=False, **common)
    model = make_model(plan, job["model"], job["options"], data, token_graphs, official_source).to(device)
    mean = torch.as_tensor(data.mean, device=device).reshape(1, 1, 1, -1)
    std = torch.as_tensor(data.std, device=device).reshape(1, 1, 1, -1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=plan["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=plan["lr_patience"])
    best, best_epoch, best_state, stale, history = math.inf, 0, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, plan["epochs"] + 1):
        tick = time.perf_counter()
        train_mae = run_epoch(model, train_loader, device, mean, std, optimizer,
                              plan["max_train_batches"], plan["grad_clip"])
        val_mae = run_epoch(model, validation_loader, device, mean, std)
        if not (math.isfinite(train_mae) and math.isfinite(val_mae)):
            raise ValueError("Non-finite training or validation error")
        scheduler.step(val_mae)
        if val_mae < best:
            best, best_epoch, stale = val_mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_mae=train_mae, val_mae=val_mae,
                            lr=float(optimizer.param_groups[0]["lr"]), seconds=time.perf_counter() - tick))
        print(f"{folder.name} epoch={epoch} train_mae={train_mae:.5f} val_mae={val_mae:.5f} "
              f"seconds={history[-1]['seconds']:.1f}", file=log, flush=True)
        if stale >= plan["patience"]:
            break
    model.load_state_dict(best_state)
    result = dict(best_val_mae=best, best_epoch=best_epoch, trained_epochs=len(history),
                  train_seconds=time.perf_counter() - started,
                  parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  test_evaluated=False)
    artifacts = {}
    with (folder / "history.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    artifacts["history.csv"] = file_hash(folder / "history.csv")
    if job["phase"] != "tuning":
        torch.save(best_state, folder / "checkpoint.pt")
        artifacts["checkpoint.pt"] = file_hash(folder / "checkpoint.pt")
        test_loader = DataLoader(test, shuffle=False, **common)
        tick = time.perf_counter()
        metrics, window_abs, window_sq = evaluate_test(model, test_loader, test, data, device,
                                                       demand_thresholds(data))
        metrics["test_seconds"] = time.perf_counter() - tick
        with (folder / "window_errors.npz").open("xb") as handle:
            np.savez_compressed(handle, abs=window_abs, sq=window_sq)
        artifacts["window_errors.npz"] = file_hash(folder / "window_errors.npz")
        result.update(test=metrics, test_evaluated=True)
    return result, artifacts


# ----------------------------------------------------------------------------
# Job graph
# ----------------------------------------------------------------------------

def jobs_for(plan):
    """All runs; finals and ablations depend on a locked validation selection."""
    jobs = []
    if "fixed_learning_rates" in plan:
        # Prespecified sensitivity runs reuse the original validation LR; no tuning/default reruns.
        for tier in plan["tiers"]:
            for variant, spec in plan["variants"].items():
                for seed in plan["final_seeds"]:
                    jobs.append(dict(tier=tier, phase="sensitivity", model=spec["model"],
                                     options=spec["options"], lr=plan["fixed_learning_rates"][tier],
                                     seed=seed, name=f"{variant}_s{seed}"))
        return jobs
    for tier in plan["tiers"]:
        for model in plan["models"]:
            for lr in plan["learning_rates"]:
                jobs.append(dict(tier=tier, phase="tuning", model=model,
                                 options=plan.get("model_options", {}).get(model, {}), lr=lr,
                                 seed=plan["tuning_seed"], name=f"{model}_lr{lr:g}"))
            for seed in plan["final_seeds"]:
                jobs.append(dict(tier=tier, phase="final", model=model,
                                 options=plan.get("model_options", {}).get(model, {}), lr=None,
                                 seed=seed, name=f"{model}_s{seed}", selection=model))
        for variant, spec in plan.get("ablations", {}).items():
            for seed in plan["final_seeds"]:
                jobs.append(dict(tier=tier, phase="ablation", model=spec["model"], options=spec["options"],
                                 lr=None, seed=seed, name=f"{variant}_s{seed}",
                                 selection=plan["ablation_selection"]))
    return jobs


def selection_path(root, tier, model):
    return root / tier / "selection" / f"{model}.json"


def try_select(root, plan, tier, model):
    """Lock the validation-minimum learning rate once all candidates finished."""
    path = selection_path(root, tier, model)
    if path.exists():
        return read_json(path)
    rows = []
    for lr in plan["learning_rates"]:
        metrics = root / tier / "tuning" / f"{model}_lr{lr:g}" / "metrics.json"
        if not metrics.exists():
            return None
        row = read_json(metrics)
        rows.append(dict(lr=lr, best_val_mae=row["best_val_mae"], metrics_sha256=file_hash(metrics)))
    chosen = min(rows, key=lambda r: (r["best_val_mae"], r["lr"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_new(path, dict(tier=tier, model=model, rule="minimum validation MAE; ties -> smaller lr",
                             candidates=rows, selected_lr=chosen["lr"], test_evaluated=False))
    except FileExistsError:
        pass
    return read_json(path)


def claim(root, job):
    lock = root / job["tier"] / "_locks" / f'{job["phase"]}__{job["name"]}'
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        return False
    (lock / "owner.json").write_text(json.dumps(dict(pid=os.getpid(), started=time.time())), encoding="utf-8")
    return True


def worker(args):
    import torch
    from data_provider.graph_tmf import load_graph_tmf
    from experiments.movement_graph import build_movement_graph, coordinates_for
    from models.tmf_v2 import proximity_token_graph
    path = plan_file(args)
    plan = read_json(path)
    execution = read_json(args.output_root / "execution.json")
    if execution["plan_sha256"] != file_hash(path):
        raise ValueError("Plan changed after preparation")
    for item in execution["sources"]:
        if file_hash(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Snapshot source changed: {item['path']}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("A worker must see exactly one CUDA device")
    torch.set_num_threads(plan["torch_threads"])
    device = torch.device("cuda")
    environment = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0), physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       platform=platform.platform())
    cache = {}

    def context(tier):
        if tier not in cache:
            spec = plan["datasets"][tier]
            path = args.data_root / "processed" / spec["filename"]
            if file_hash(path) != spec["sha256"]:
                raise ValueError("Dataset bytes changed")
            data = load_graph_tmf(path)
            names = list(data.names)
            movement, _, _ = build_movement_graph(names, coordinates_for(names, args.metadata))
            graphs = dict(movement=movement, proximity=proximity_token_graph(data.adj, data.n_channels))
            cache[tier] = (data, calendar_indices(data.dates), graphs)
        return cache[tier]

    log = sys.stdout
    while True:
        jobs = jobs_for(plan)
        pending = False
        progressed = False
        for job in jobs:
            folder = args.output_root / job["tier"] / job["phase"] / job["name"]
            if (folder / "metrics.json").exists():
                continue
            pending = True
            if "selection" in job:
                selection = try_select(args.output_root, plan, job["tier"], job["selection"])
                if selection is None:
                    continue
                job = dict(job, lr=selection["selected_lr"],
                           selection_sha256=file_hash(selection_path(args.output_root, job["tier"], job["selection"])))
            heavy_lock = None
            if job["model"] in plan["heavy_models"]:
                # At most one ~20 GB model per physical GPU at a time.
                heavy_lock = args.output_root / "_heavy" / f'gpu{os.environ.get("CUDA_VISIBLE_DEVICES")}'
                heavy_lock.parent.mkdir(parents=True, exist_ok=True)
                try:
                    heavy_lock.mkdir()
                except FileExistsError:
                    continue
            if not claim(args.output_root, job):
                if heavy_lock is not None:
                    heavy_lock.rmdir()
                continue
            progressed = True
            if folder.exists():
                target = args.output_root / "_interrupted" / f'{job["tier"]}__{job["phase"]}__{job["name"]}__{int(time.time())}'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(folder), str(target))
                print("MOVED INTERRUPTED", folder, "->", target, flush=True)
            folder.mkdir(parents=True)
            identity = dict(job, plan_sha256=execution["plan_sha256"],
                            source_sha256=execution["source_sha256"])
            write_new(folder / "config.json", identity)
            data, cal, graphs = context(job["tier"])
            print("START", job["tier"], job["phase"], job["name"], "lr", job["lr"], flush=True)
            try:
                result, artifacts = fit(plan, job, data, cal, graphs, args.staeformer_source, device, folder, log)
            finally:
                if heavy_lock is not None:
                    heavy_lock.rmdir()
                torch.cuda.empty_cache()
            artifacts["config.json"] = file_hash(folder / "config.json")
            write_new(folder / "metrics.json", dict(identity, **result, status="completed",
                                                    data_sha256=data.fingerprint,
                                                    environment=environment, artifacts=artifacts))
            print("COMPLETED", job["tier"], job["phase"], job["name"], flush=True)
            if job["phase"] == "tuning":
                try_select(args.output_root, plan, job["tier"], job["model"])
            break
        if not pending:
            print("NO PENDING JOBS", flush=True)
            return
        if not progressed:
            time.sleep(30)


# ----------------------------------------------------------------------------
# Preparation and supervision
# ----------------------------------------------------------------------------

def prepare(args):
    path = plan_file(args)
    plan = read_json(path)
    locks = {}
    if plan.get("verify_fixed_learning_rate_locks"):
        selection_root = getattr(args, "selection_root", None)
        if selection_root is None:
            raise ValueError("This plan requires --selection-root pointing to the original benchmark_v2 results")
        for tier in plan["tiers"]:
            lock = Path(selection_root) / tier / "selection/st_itransformer.json"
            value = read_json(lock)
            if (value.get("tier") != tier or value.get("model") != "st_itransformer"
                    or value.get("test_evaluated") is not False
                    or value.get("selected_lr") != plan["fixed_learning_rates"][tier]):
                raise ValueError(f"Fixed LR disagrees with original validation selection lock: {tier}")
            locks[f"selection_locks/{tier}/st_itransformer.json"] = lock
    for tier, spec in plan["datasets"].items():
        if file_hash(args.data_root / "processed" / spec["filename"]) != spec["sha256"]:
            raise ValueError(f"Dataset {tier} differs from the plan")
    source = args.output_root / "source"
    if source.exists():
        raise ValueError("Snapshot exists; inspect it instead of overwriting")
    files = snapshot_files(path)
    for name in files:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    for name, original in locks.items():
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    files += list(locks)
    sources = [dict(path=name, sha256=file_hash(source / name)) for name in files]
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    write_new(source / "execution.json", dict(plan_sha256=file_hash(source / path.relative_to(ROOT)),
                                              sources=sources, source_sha256=digest,
                                              prepared=datetime.now().isoformat(timespec="seconds")))
    shutil.copy2(source / "execution.json", args.output_root / "execution.json")
    print("SNAPSHOT", source, "SOURCE SHA256", digest, flush=True)


def pipeline(args):
    source = args.output_root / "source"
    execution = read_json(args.output_root / "execution.json")
    selected_plan = plan_file(args).relative_to(ROOT)
    if file_hash(source / selected_plan) != execution["plan_sha256"]:
        raise ValueError("Selected snapshot plan differs from preparation")
    receipt = dict(status="running", source_sha256=execution["source_sha256"], workers=[])
    processes = []
    for gpu in args.gpus:
        for slot in range(args.workers_per_gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUTF8="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            log_path = args.output_root / f"worker_gpu{gpu}_{slot}.log"
            command = [sys.executable, "-B", "-u", "-m", "experiments.benchmark_v2", "worker",
                       "--plan", selected_plan.as_posix(),
                       "--data-root", str(args.data_root), "--output-root", str(args.output_root),
                       "--metadata", str(args.metadata), "--staeformer-source", str(args.staeformer_source)]
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((process, handle))
            receipt["workers"].append(dict(gpu=gpu, slot=slot, pid=process.pid, log=log_path.name, exit_code=None))
            time.sleep(20)
    path = args.output_root / "pipeline.json"
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    while any(row["exit_code"] is None for row in receipt["workers"]):
        for (process, handle), row in zip(processes, receipt["workers"]):
            if row["exit_code"] is None and process.poll() is not None:
                row["exit_code"] = process.returncode
                handle.close()
        receipt["status"] = "running_with_failure" if any(
            row["exit_code"] not in (None, 0) for row in receipt["workers"]) else "running"
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        time.sleep(30)
    receipt["status"] = "completed" if all(row["exit_code"] == 0 for row in receipt["workers"]) else "failed"
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "pipeline", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, default=PLAN,
                       help="Plan under the source root (default: benchmark_v2.json)")
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--output-root", type=Path, required=True)
        if name != "prepare":
            p.add_argument("--metadata", type=Path, required=True)
            p.add_argument("--staeformer-source", type=Path, required=True)
        else:
            p.add_argument("--selection-root", type=Path,
                           help="Original benchmark_v2 results; required by fixed-LR controls")
        if name == "pipeline":
            p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
            p.add_argument("--workers-per-gpu", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        args.output_root.mkdir(parents=True, exist_ok=True)
        prepare(args)
    elif args.command == "pipeline":
        pipeline(args)
    else:
        with redirect_stderr(sys.stdout):
            worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
