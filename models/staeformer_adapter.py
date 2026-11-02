"""Flow-only TMF adapter around UNMODIFIED, externally supplied STAEformer.

No upstream file is redistributed here: the inspected official repository has
no root license. Runtime source must match the reviewed pinned commit's bytes.
The calendar branches are explicitly off; this is not a full-calendar baseline
or a reproduction of upstream dataset/loss/optimizer benchmark protocols.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn

UPSTREAM_COMMIT = 'fc49d39b2f1a8e3cf37b6289d7240680e1690f3f'
UPSTREAM_URL = f'https://raw.githubusercontent.com/XDZhelheim/STAEformer/{UPSTREAM_COMMIT}/model/STAEformer.py'
UPSTREAM_SHA256 = 'fb2292ae1a1259131fafb16392c6b76097d16bc7fb1c36c6d3ce97ebe7be76c7'


def load_official_class(source_path: str | Path):
    source = Path(source_path).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError('STAEformer bytes differ from reviewed official commit')
    spec = importlib.util.spec_from_file_location('tmf_pinned_external_staeformer', source)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load external pinned STAEformer source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAEformer


class STAEformerNoCalendarForecaster(nn.Module):
    """Physical intersections are spatial nodes; 12 movements are input/output channels.

Temporal and spatial attention, adaptive embedding, and output projection are
all upstream code. No RevIN, context concatenation, graph mask, or extra identity
is added. The input/output dimensions are a multivariate TMF task adaptation.
"""
    def __init__(self, seq_len: int, pred_len: int, n_nodes: int, *,
                 official_source: str | Path, input_dim: int = 12,
                 adaptive_embedding_dim: int = 80):
        super().__init__()
        for value in (seq_len, pred_len, n_nodes, input_dim, adaptive_embedding_dim):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('STAEformer dimensions must be positive integers')
        if (24 + adaptive_embedding_dim) % 4:
            raise ValueError('Feature width must be divisible by four attention heads')
        self.seq_len, self.pred_len, self.n_nodes, self.input_dim = seq_len, pred_len, n_nodes, input_dim
        self.official_settings = dict(num_nodes=n_nodes, in_steps=seq_len, out_steps=pred_len,
            steps_per_day=288, input_dim=input_dim, output_dim=input_dim,
            input_embedding_dim=24, tod_embedding_dim=0, dow_embedding_dim=0,
            spatial_embedding_dim=0, adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1, use_mixed_proj=True)
        self.official = load_official_class(official_source)(**self.official_settings)

    def forward(self, x: torch.Tensor, adjacency=None, time_features=None):
        if time_features is not None:
            raise ValueError('Calendar features cannot be silently discarded by a flow-only baseline')
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.seq_len, self.n_nodes, self.input_dim):
            raise ValueError('Expected physical-node multivariate TMF [B,L,N,C] input')
        return self.official(x)
