# ST-iTransformer

Core code for network-level turning movement flow (TMF) forecasting with ST-iTransformer:
movement-level variable tokens that carry demand-level context, a movement corridor graph, and the
chronological multi-horizon benchmark used in the accompanying IEEE T-ITS manuscript.

The Milton, Ontario, traffic data and trained weights are not distributed. The code runs on any
dataset in the format below.

## Contents

```text
configs/experiments/   benchmark plans (main comparison, stronger baselines, controls, sensitivity, budget)
data_provider/         dataset loading and chronological windows
models/tmf_v2.py       ST-iTransformer and every baseline behind one interface
models/vendor/         unmodified DLinear, STID and BasicTS (STGCN, AGCRN, Graph WaveNet) code
experiments/           benchmark_v2.py (tuning, selection, training, evaluation), movement_graph.py
tests/test_core.py     CPU checks on synthetic data
```

## Install

Python 3.10+ and PyTorch.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## External baseline sources

These are loaded at run time and checked against SHA-256 hashes; they are not redistributed here.

- STAEformer: `model/STAEformer.py` from https://github.com/XDZhelheim/STAEformer at commit
  `fc49d39b2f1a8e3cf37b6289d7240680e1690f3f`; pass its path with `--staeformer-source`.
- HimNet: `HimNet.py`, `HimNet.yaml` and `HimNetRunner.py` from https://github.com/XDZhelheim/HimNet
  at commit `2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5`, copied into `external/himnet/`.

## Data format

`<data-root>/processed/<filename>.npz`, named in the plan's `datasets` entry together with its SHA-256:

| key | shape / type | meaning |
|---|---|---|
| `x` | `[T, N, 12]` float | five-minute movement counts, scaled with training statistics |
| `mean`, `std` | broadcastable to `x` | training-only scaling, used to restore vehicle counts |
| `dates` | `[T]` str | timestamps `YYYY-MM-DD HH:MM:SS` |
| `names` | `[N]` str | intersection names `Street A and Street B` |
| `target_columns` | `[12]` str | movement order, `EBL, EBR, EBT, NBL, ..., WBT` |
| `adj` | `[N, N]` float | intersection adjacency |
| `target`, `train_end`, `val_end` | scalars | target label and chronological 70/10/20 split indices |

The corridor graph is built from `--metadata`, a JSON list of `{"name", "lat", "lon"}` records for
the intersections in `names`.

## Run the benchmark

```bash
python -m experiments.benchmark_v2 prepare  --data-root DATA --output-root OUT
python -m experiments.benchmark_v2 pipeline --data-root DATA --output-root OUT \
    --metadata intersections.json --staeformer-source STAEformer.py --gpus 0
```

`prepare` freezes a hashed source snapshot under `OUT/source/`; `pipeline` tunes learning rates on
validation, locks the selection, and trains three final seeds per model. Other plans are selected with
`--plan configs/experiments/<plan>.json`.

## License

MIT (see `LICENSE`). Files under `models/vendor/` keep their original licenses.
