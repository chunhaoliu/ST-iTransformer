"""CPU checks on synthetic data: corridor graph, protocol, evaluation and models."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data_provider.graph_tmf import GraphTMFData
from experiments import benchmark_v2 as bench
from experiments.movement_graph import (APPROACH, CHANNELS, DISCHARGE, build_movement_graph,
                                        orientation_check, street_key)
from models import tmf_v2
from models.himnet_adapter import SOURCE as HIMNET

ROOT = Path(__file__).resolve().parents[1]
STAEFORMER = Path(os.environ.get("TMF_STAEFORMER_SOURCE", str(ROOT / "external/STAEformer.py")))
# External sources are supplied by the user (README); D2STGNN was excluded from the study.
SKIP = {"d2stgnn"} | ({"staeformer"} if not STAEFORMER.exists() else set())
SKIP |= {"himnet"} if not (HIMNET / "HimNet.py").exists() else set()


def synthetic(n_steps=400, n_nodes=3, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.poisson(3.0, size=(n_steps, n_nodes, 12)).astype(np.float32)
    raw[:, :, 1] = 0.0  # an always-zero movement
    train_end, val_end = int(0.7 * n_steps), int(0.8 * n_steps)
    mean = raw[:train_end].mean(axis=(0, 1), keepdims=True)
    std = raw[:train_end].std(axis=(0, 1), keepdims=True)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    dates = np.array([str(np.datetime64("2021-03-01T00:00") + np.timedelta64(5 * i, "m")).replace("T", " ")
                      for i in range(n_steps)])
    data = GraphTMFData(x=((raw - mean) / std).astype(np.float32), adj=np.ones((n_nodes, n_nodes), np.float32),
                        mean=mean.astype(np.float32), std=std, dates=dates,
                        names=np.array([f"Main Street and Road {i}" for i in range(n_nodes)]),
                        target_columns=np.array(CHANNELS), target="TMF12", train_end=train_end,
                        val_end=val_end, source_path=Path("synthetic"), fingerprint="synthetic")
    return data, raw


class MovementGraphTests(unittest.TestCase):
    def test_street_names_normalise_compass_suffixes(self):
        self.assertEqual(street_key("Main Street E"), street_key("Main Street East"))
        self.assertEqual(street_key("Thompson Road North"), street_key("Thompson Road"))

    def test_eastbound_discharge_feeds_downstream_eastbound_approach(self):
        # Two intersections on Main Street; B lies to the true north-east of A,
        # which is nominal east on Milton's rotated grid.
        names = ["Main Street and Alpha Road", "Main Street and Beta Road"]
        coords = np.array([[43.5000, -79.9000], [43.5050, -79.8930]])
        graph, links, axes = build_movement_graph(names, coords)
        self.assertEqual(axes["main street"], "EW")
        channel = {c: i for i, c in enumerate(CHANNELS)}
        a, b = 0, 12
        for src in DISCHARGE["E"]:
            for dst in APPROACH["E"]:
                self.assertTrue(graph[b + channel[dst], a + channel[src]])
        # Westbound movements at A do not feed B, and B's westbound discharge feeds A.
        self.assertFalse(graph[b + channel["EBT"], a + channel["WBT"]])
        self.assertTrue(graph[a + channel["WBT"], b + channel["WBT"]])
        self.assertTrue(graph[:12, :12].all() and graph[12:, 12:].all())
        self.assertEqual({l["direction"] for l in links}, {"E", "W"})

    def test_orientation_check_prefers_coupled_approach(self):
        names = ["Main Street and Alpha Road", "Main Street and Beta Road"]
        coords = np.array([[43.5000, -79.9000], [43.5050, -79.8930]])
        _, links, _ = build_movement_graph(names, coords)
        rng = np.random.default_rng(1)
        raw = rng.poisson(5.0, size=(2000, 2, 12)).astype(float)
        channel = {c: i for i, c in enumerate(CHANNELS)}
        discharge = raw[:, 0, [channel[c] for c in DISCHARGE["E"]]].sum(axis=1)
        raw[:, 1, channel["EBT"]] = discharge  # same-interval arrivals
        result = orientation_check(raw, links, names)
        eastbound = [r for r in result["rows"] if r["direction"] == "E"][0]
        self.assertGreater(eastbound["matched_corr"], eastbound["opposite_corr"])


class ProtocolTests(unittest.TestCase):
    def test_calendar_indices(self):
        values = bench.calendar_indices(["2021-03-01 13:25:00", "2022-12-31 23:55:00"])
        self.assertEqual(values.tolist(), [[13 * 12 + 5, 0, 2], [287, 5, 11]])

    def test_windows_keep_targets_inside_their_split(self):
        data, _ = synthetic()
        train, validation, test = bench.split_windows(data, bench.calendar_indices(data.dates), 12, 12)
        self.assertLess(train.target_positions().max(), data.train_end)
        self.assertGreaterEqual(validation.target_positions().min(), data.train_end)
        self.assertLess(validation.target_positions().max(), data.val_end)
        self.assertGreaterEqual(test.target_positions().min(), data.val_end)
        self.assertEqual(test.target_positions().max(), len(data.x) - 1)

    def test_demand_thresholds_use_training_rows_only(self):
        data, raw = synthetic()
        before = bench.demand_thresholds(data)
        changed = data.x.copy()
        changed[data.train_end:] = 1e6
        altered = GraphTMFData(**{**data.__dict__, "x": changed})
        np.testing.assert_allclose(before, bench.demand_thresholds(altered))
        self.assertTrue(np.isinf(before[:, 1]).all())

    def test_evaluation_is_exact_for_an_oracle_and_clips_negative_forecasts(self):
        data, raw = synthetic()
        cal = bench.calendar_indices(data.dates)
        _, _, test = bench.split_windows(data, cal, 12, 12)
        loader = torch.utils.data.DataLoader(test, batch_size=7)

        positions = test.target_positions()
        truth = torch.as_tensor(data.x[positions])  # [W, H, N, C]

        class Lookup(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset
                self.calls = 0

            def forward(self, x, cal):
                start = self.calls
                self.calls += x.shape[0]
                return truth[start:start + x.shape[0]] + self.offset

        metrics, window_abs, _ = bench.evaluate_test(Lookup(0.0), loader, test, data, torch.device("cpu"),
                                                     bench.demand_thresholds(data))
        self.assertAlmostEqual(metrics["mae_all"], 0.0, places=5)
        self.assertEqual(window_abs.shape, (len(test), 12))
        self.assertEqual(metrics["targets"], len(test) * 12 * 3 * 12)
        counts = sum(group["count"] for group in metrics["demand"].values())
        self.assertEqual(counts, metrics["targets"])
        self.assertEqual(sum(m["count"] for m in metrics["months"].values()), metrics["targets"])
        # A large negative offset is clipped at zero vehicles: error equals the true count.
        offset = -1e3
        metrics, _, _ = bench.evaluate_test(Lookup(offset), loader, test, data, torch.device("cpu"),
                                            bench.demand_thresholds(data))
        expected = raw[positions].mean()
        self.assertAlmostEqual(metrics["mae_all"], float(expected), places=3)
        self.assertGreater(metrics["mae_unclipped_all"], metrics["mae_all"])

    def test_selection_locks_validation_minimum(self):
        plan = json.loads((ROOT / "configs/experiments/benchmark_v2.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIsNone(bench.try_select(root, plan, "core20", "stid"))
            for lr, value in zip(plan["learning_rates"], (0.5, 0.4, 0.4)):
                path = root / "core20/tuning" / f"stid_lr{lr:g}"
                path.mkdir(parents=True)
                (path / "metrics.json").write_text(json.dumps(dict(best_val_mae=value)), encoding="utf-8")
            selection = bench.try_select(root, plan, "core20", "stid")
            self.assertEqual(selection["selected_lr"], 0.001)
            self.assertFalse(selection["test_evaluated"])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        names = ["Main Street and Alpha Road", "Main Street and Beta Road", "Main Street and Gamma Road"]
        coords = np.array([[43.500, -79.900], [43.505, -79.893], [43.510, -79.886]])
        self.graph, _, _ = build_movement_graph(names, coords)
        self.x = torch.randn(2, 12, 3, 12)
        self.cal = torch.stack([torch.randint(0, 288, (2, 12)), torch.randint(0, 7, (2, 12)),
                                torch.randint(0, 12, (2, 12))], dim=-1)


    def test_every_model_maps_to_horizon_shape_and_backpropagates(self):
        for name in tmf_v2.MODELS:
            if name in SKIP:
                continue
            with self.subTest(model=name):
                model = tmf_v2.build(name, seq_len=12, pred_len=12, n_nodes=3, token_graph=self.graph,
                                     official_source=STAEFORMER)
                if getattr(model, "requires_future_calendar", False):
                    # Synthetic target labels; no target values enter evaluation.
                    cal = torch.cat((self.cal, self.cal), dim=1)
                    out = model.training_forward(self.x, cal, torch.zeros_like(self.x))
                else:
                    out = model(self.x, self.cal)
                self.assertEqual(tuple(out.shape), (2, 12, 3, 12))
                out.square().mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    @unittest.skipUnless(STAEFORMER.exists(), "pinned official STAEformer source not available")
    def test_staeformer_forward_matches_official(self):
        model = tmf_v2.build("staeformer", seq_len=12, pred_len=12, n_nodes=3, n_channels=1,
                             official_source=STAEFORMER).eval()
        x = self.x[..., :1]
        tod = ((self.cal[..., 0].float() + 0.5) / 288.0)[:, :, None, None].expand(-1, -1, 3, 1)
        dow = self.cal[..., 1].float()[:, :, None, None].expand(-1, -1, 3, 1)
        official = model.official(torch.cat((x, tod, dow), dim=-1))
        torch.testing.assert_close(model(x, self.cal), official)

    def test_refinement_controls_match_capacity(self):
        counts = {}
        for refinement in ("movement", "dense", "permuted", "proximity"):
            graph = tmf_v2.proximity_token_graph(np.ones((3, 3)), 12) if refinement == "proximity" else self.graph
            model = tmf_v2.build("st_itransformer", seq_len=12, pred_len=12, n_nodes=3, token_graph=graph,
                                 refinement=refinement)
            counts[refinement] = sum(p.numel() for p in model.parameters())
        self.assertEqual(len(set(counts.values())), 1)

    def test_movement_mask_blocks_unlinked_tokens(self):
        model = tmf_v2.build("st_itransformer", seq_len=12, pred_len=12, n_nodes=3, token_graph=self.graph)
        allowed = ~model.blocked.numpy()
        np.testing.assert_array_equal(allowed, self.graph | np.eye(36, dtype=bool))
        dense = tmf_v2.build("st_itransformer", seq_len=12, pred_len=12, n_nodes=3, token_graph=self.graph,
                             refinement="dense")
        self.assertFalse(dense.blocked.any())

    def test_calendar_changes_output_only_when_enabled(self):
        shifted = self.cal.clone()
        shifted[..., 0] = (shifted[..., 0] + 100) % 288
        for enabled in (True, False):
            model = tmf_v2.build("st_itransformer", seq_len=12, pred_len=12, n_nodes=3, token_graph=self.graph,
                                 use_calendar=enabled).eval()
            same = torch.allclose(model(self.x, self.cal), model(self.x, shifted))
            self.assertEqual(same, not enabled)

    def test_one_cpu_training_run(self):
        data, _ = synthetic(n_steps=300)
        plan = json.loads((ROOT / "configs/experiments/benchmark_v2.json").read_text(encoding="utf-8"))
        plan = dict(plan, epochs=2, max_train_batches=3, batch_size=8)
        job = dict(tier="core20", phase="final", model="st_itransformer", options={}, lr=1e-3, seed=2026,
                   name="st_itransformer_s2026")
        graphs = dict(movement=build_movement_graph(list(data.names), np.array(
            [[43.500, -79.900], [43.505, -79.893], [43.510, -79.886]]))[0])
        with tempfile.TemporaryDirectory() as folder:
            result, artifacts = bench.fit(plan, job, data, bench.calendar_indices(data.dates), graphs, None,
                                          torch.device("cpu"), Path(folder), io.StringIO())
            self.assertTrue(result["test_evaluated"])
            self.assertEqual(set(artifacts), {"history.csv", "checkpoint.pt", "window_errors.npz"})
            self.assertEqual(len(result["test"]["mae"]), 12)


if __name__ == "__main__":
    unittest.main()
