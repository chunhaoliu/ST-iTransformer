"""Thin TMF interfaces for pinned official models, not reimplemented lookalikes.

Pending runtime validation: intentionally not in the experiment registry yet.
Original model bytes and licenses are retained in models/vendor/. The local
adapter is new code. STID-no-calendar is an explicitly restricted variant, not
the full temporal-identity baseline. Do not report performance before training.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .vendor.dlinear import Model as OfficialDLinear
from .vendor.stid_arch import STID as OfficialSTID


class _TMFInterface(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, n_nodes: int, input_dim: int) -> None:
        super().__init__()
        if any(int(v) != v or v <= 0 for v in (seq_len, pred_len, n_nodes, input_dim)):
            raise ValueError("Window, horizon, node and movement dimensions must be positive integers")
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.n_nodes, self.input_dim = int(n_nodes), int(input_dim)

    def _series(self, x: torch.Tensor, time_features: torch.Tensor | None) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.seq_len, self.n_nodes, self.input_dim):
            raise ValueError("Expected x with shape [B,seq_len,n_nodes,input_dim]")
        if time_features is not None:
            raise ValueError("This adapter is explicitly no-calendar; do not silently discard covariates")
        return x.reshape(x.shape[0], self.seq_len, self.n_nodes * self.input_dim)


class DLinearForecaster(_TMFInterface):
    """Official DLinear with its original 25-point endpoint-padded decomposition.

    The short L=6 setting remains legal because the official moving average
    repeats endpoints. Kernel 25 is retained, not selected on the test set.
    """
    def __init__(self, seq_len: int, pred_len: int, n_nodes: int, input_dim: int = 12,
                 individual: bool = False) -> None:
        super().__init__(seq_len, pred_len, n_nodes, input_dim)
        if not isinstance(individual, bool):
            raise ValueError("individual must be an explicit boolean")
        self.individual = individual
        self.official = OfficialDLinear(SimpleNamespace(
            seq_len=self.seq_len, pred_len=self.pred_len,
            enc_in=self.n_nodes * self.input_dim, individual=individual,
        ))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None,
                time_features: torch.Tensor | None = None) -> torch.Tensor:
        series = self._series(x, time_features)
        return self.official(series).reshape(x.shape[0], self.pred_len, self.n_nodes, self.input_dim)


class STIDNoCalendarForecaster(_TMFInterface):
    """Original STID model with calendar branches deliberately disabled.

    Node-movement histories are its N*C separate series identities, matching
    the variate interface used by iTransformer. No extra future information,
    graph, per-window standardization or dual context is added by this adapter.
    Defaults follow the upstream widths/layers; MLP dropout stays at 0.15.
    """
    def __init__(self, seq_len: int, pred_len: int, n_nodes: int, input_dim: int = 12,
                 embed_dim: int = 32, node_dim: int = 32, num_layer: int = 3) -> None:
        super().__init__(seq_len, pred_len, n_nodes, input_dim)
        if any(int(v) != v or v <= 0 for v in (embed_dim, node_dim, num_layer)):
            raise ValueError("STID widths and layer count must be positive integers")
        self.official_settings = dict(
            num_nodes=self.n_nodes * self.input_dim, input_len=self.seq_len,
            input_dim=1, embed_dim=int(embed_dim), output_len=self.pred_len,
            num_layer=int(num_layer), if_node=True, node_dim=int(node_dim),
            if_T_i_D=False, if_D_i_W=False, temp_dim_tid=32, temp_dim_diw=32,
            time_of_day_size=288, day_of_week_size=7,
        )
        self.official = OfficialSTID(**self.official_settings)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None,
                time_features: torch.Tensor | None = None) -> torch.Tensor:
        series = self._series(x, time_features).unsqueeze(-1)
        prediction = self.official(series, None, batch_seen=0, epoch=0, train=self.training)
        return prediction.reshape(x.shape[0], self.pred_len, self.n_nodes, self.input_dim)
