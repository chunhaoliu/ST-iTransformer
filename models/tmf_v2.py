"""Multi-horizon TMF models sharing one interface for the v2 benchmark.

Every model maps ``x`` [B, L, N, C] in training-normalized units and the
calendar of the observed steps ``cal`` [B, L, 3] (five-minute slot 0..287,
weekday 0..6 and month 0..11, integers taken from the observed source labels) to a
forecast [B, H, N, C] in the same units.  Only observed inputs are used.

Node-based published baselines treat each intersection-movement series as one
graph node (N*C nodes, one value channel), which is the setting their
official code is written for; calendar channels follow each model's own
convention.  Official sources stay unmodified under ``models/vendor``.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

VENDOR = Path(__file__).resolve().parent / "vendor"


# ----------------------------------------------------------------------------
# Graph helpers
# ----------------------------------------------------------------------------

def proximity_token_graph(adj: np.ndarray, n_channels: int) -> np.ndarray:
    """Expand the node proximity mask (plus self-links) to all movement pairs."""
    graph = (np.asarray(adj) > 0) | np.eye(len(adj), dtype=bool)
    return np.kron(graph, np.ones((n_channels, n_channels), dtype=bool))


def permute_token_graph(graph: np.ndarray, n_nodes: int, n_channels: int, seed: int = 314159) -> np.ndarray:
    """Relabel intersections with one fixed permutation, keeping graph structure."""
    order = np.random.default_rng(seed).permutation(n_nodes)
    tokens = (order[:, None] * n_channels + np.arange(n_channels)[None, :]).reshape(-1)
    return graph[np.ix_(tokens, tokens)]


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    degree = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, degree, out=np.zeros_like(matrix), where=degree > 0)


def node_supports(token_graph: np.ndarray) -> dict[str, np.ndarray]:
    """Supports for node-based baselines from the query-attends-key graph.

    ``token_graph[q, k]`` means key k informs query q, i.e. an edge k -> q;
    ``flow[v, w]`` is the conventional source-to-target adjacency.
    """
    flow = token_graph.T.astype(np.float32)
    symmetric = np.maximum(flow, flow.T)
    degree = symmetric.sum(axis=1)
    inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)
    laplacian = np.eye(len(flow), dtype=np.float32) - inv_sqrt[:, None] * symmetric * inv_sqrt[None, :]
    return dict(forward=row_normalize(flow).astype(np.float32),
                backward=row_normalize(flow.T).astype(np.float32),
                laplacian=laplacian.astype(np.float32))


# ----------------------------------------------------------------------------
# ST-iTransformer (v2)
# ----------------------------------------------------------------------------

class STiTransformerV2(nn.Module):
    """Level-, identity- and time-aware inverted tokens with movement refinement.

    Each intersection-movement history is one token.  Its input interleaves
    window-standardized variation with the observed training-normalized level.
    Learned intersection and movement identities and a time-of-day/day-of-week
    embedding of the forecast origin condition the token.  After global
    attention, one masked refinement layer lets a token attend the movements
    of its own intersection and the upstream movements that discharge into
    its approach; a learned gate fuses global and refined states.  The window
    mean and scale are restored at the output, as in iTransformer.

    Opt-in controls replace level history with shape/mean/scale features,
    remove window normalization, split identity/calendar inputs, restrict the
    encoder to diagonal attention, or restrict refinement to within-site links.
    """

    REFINEMENTS = ("movement", "proximity", "dense", "permuted", "none", "within_intersection")

    def __init__(self, seq_len: int, pred_len: int, n_nodes: int, n_channels: int = 12, *,
                 token_graph: np.ndarray | None = None, d_model: int = 128, n_heads: int = 8,
                 e_layers: int = 2, d_ff: int = 256, dropout: float = 0.05,
                 use_context: bool = True, use_identity: bool = True, use_calendar: bool = True,
                 refinement: str = "movement", use_fusion: bool = True,
                 input_mode: str = "level_history", cross_attention: bool = True,
                 use_intersection_id: bool = True, use_movement_id: bool = True,
                 use_time_of_day: bool = True, use_weekday: bool = True, **_: object) -> None:
        super().__init__()
        if refinement not in self.REFINEMENTS:
            raise ValueError(f"refinement must be one of {self.REFINEMENTS}")
        if refinement != "none" and token_graph is None:
            raise ValueError("A token graph is required for masked refinement")
        self.seq_len, self.pred_len = seq_len, pred_len
        self.n_nodes, self.n_channels = n_nodes, n_channels
        self.use_context, self.use_identity, self.use_calendar = use_context, use_identity, use_calendar
        if input_mode not in ("level_history", "stats_restoration", "no_window_norm"):
            raise ValueError("Unknown input_mode")
        if not cross_attention and (refinement != "none" or use_fusion):
            raise ValueError("Token-local control requires refinement='none', use_fusion=False")
        self.input_mode, self.cross_attention = input_mode, cross_attention
        self.use_intersection_id, self.use_movement_id = use_intersection_id, use_movement_id
        self.use_time_of_day, self.use_weekday = use_time_of_day, use_weekday
        self.refinement, self.use_fusion = refinement, use_fusion and refinement != "none"
        width = seq_len * (2 if use_context else 1)
        if input_mode == "stats_restoration":
            width = seq_len + 2  # standardized shape, window mean, window scale
        elif input_mode == "no_window_norm":
            width = seq_len
        self.embedding = nn.Linear(width, d_model)
        if use_identity:
            if use_intersection_id:
                self.node_identity = nn.Parameter(torch.randn(n_nodes, 1, d_model) * 0.02)
            if use_movement_id:
                self.movement_identity = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        if use_calendar:
            if use_time_of_day:
                self.time_of_day = nn.Embedding(288, d_model)
            if use_weekday:
                self.day_of_week = nn.Embedding(7, d_model)
            if use_time_of_day:
                nn.init.normal_(self.time_of_day.weight, std=0.02)
            if use_weekday:
                nn.init.normal_(self.day_of_week.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        if not cross_attention:
            self.register_buffer("encoder_blocked", ~torch.eye(n_nodes * n_channels, dtype=torch.bool),
                                 persistent=False)
        if refinement != "none":
            tokens = n_nodes * n_channels
            graph = np.asarray(token_graph, dtype=bool)
            if graph.shape != (tokens, tokens):
                raise ValueError("token_graph must have shape [N*C, N*C]")
            if refinement == "proximity" or refinement == "movement":
                allowed = graph
            elif refinement == "dense":
                allowed = np.ones_like(graph)
            elif refinement == "within_intersection":
                allowed = np.kron(np.eye(n_nodes, dtype=bool), np.ones((n_channels, n_channels), dtype=bool))
            else:
                allowed = permute_token_graph(graph, n_nodes, n_channels)
            allowed = allowed | np.eye(tokens, dtype=bool)
            self.register_buffer("blocked", torch.as_tensor(~allowed), persistent=False)
            self.refine = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.refine_norm = nn.LayerNorm(d_model)
            self.refine_dropout = nn.Dropout(dropout)
            if self.use_fusion:
                self.gate = nn.Linear(2 * d_model, d_model)
                nn.init.zeros_(self.gate.bias)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor, cal: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, nodes, channels = x.shape
        if (length, nodes, channels) != (self.seq_len, self.n_nodes, self.n_channels):
            raise ValueError("Unexpected input shape")
        if self.input_mode == "no_window_norm":
            mean, scale, standardized = None, None, x
        else:
            mean = x.mean(dim=1, keepdim=True).detach()
            scale = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            standardized = (x - mean) / scale
        series = standardized.permute(0, 2, 3, 1)  # [B, N, C, L]
        if self.use_context and self.input_mode == "level_history":
            series = torch.stack((series, x.permute(0, 2, 3, 1)), dim=-1).flatten(-2)
        if self.input_mode == "stats_restoration":
            series = torch.cat((standardized.permute(0, 2, 3, 1), mean[:, 0, :, :, None],
                                scale[:, 0, :, :, None]), dim=-1)
        elif self.input_mode == "no_window_norm":
            series = x.permute(0, 2, 3, 1)
        tokens = self.embedding(series)  # [B, N, C, d]
        if self.use_identity:
            # Preserve the original addition order and parameter initialization.
            if self.use_intersection_id:
                tokens = tokens + self.node_identity
            if self.use_movement_id:
                tokens = tokens + self.movement_identity
        if self.use_calendar:
            if cal is None:
                raise ValueError("Calendar input required")
            origin = cal[:, -1].long()
            time = (self.time_of_day(origin[:, 0]) if self.use_time_of_day else 0) + (
                self.day_of_week(origin[:, 1]) if self.use_weekday else 0)
            if not self.use_time_of_day and not self.use_weekday:
                time = tokens.new_zeros((batch, tokens.shape[-1]))
            tokens = tokens + time[:, None, None, :]
        tokens = tokens.reshape(batch, nodes * channels, -1)
        encoded = self.encoder(tokens) if self.cross_attention else self.encoder(tokens, mask=self.encoder_blocked)
        if self.refinement != "none":
            refined, _ = self.refine(encoded, encoded, encoded, attn_mask=self.blocked, need_weights=False)
            refined = self.refine_norm(encoded + self.refine_dropout(refined))
            if self.use_fusion:
                gate = torch.sigmoid(self.gate(torch.cat((encoded, refined), dim=-1)))
                encoded = gate * refined + (1.0 - gate) * encoded
            else:
                encoded = refined
        forecast = self.head(encoded).reshape(batch, nodes, channels, self.pred_len).permute(0, 3, 1, 2)
        return forecast if self.input_mode == "no_window_norm" else forecast * scale + mean


# ----------------------------------------------------------------------------
# Flow-only and recurrent comparators
# ----------------------------------------------------------------------------

class ITransformerV2(nn.Module):
    """Ordinary iTransformer: standardized variate tokens, no identities or calendar."""

    def __init__(self, seq_len: int, pred_len: int, n_nodes: int, n_channels: int = 12, **_: object):
        super().__init__()
        from .i_transformer import ITransformerForecaster
        self.inner = ITransformerForecaster(seq_len=seq_len, pred_len=pred_len, n_nodes=n_nodes,
                                            input_dim=n_channels)

    def forward(self, x, cal=None):
        return self.inner(x, None)


class DLinearV2(nn.Module):
    """Official DLinear (shared head) over the flattened movement series."""

    def __init__(self, seq_len: int, pred_len: int, n_nodes: int, n_channels: int = 12, **_: object):
        super().__init__()
        from .modern_baselines import DLinearForecaster
        self.inner = DLinearForecaster(seq_len, pred_len, n_nodes, n_channels, individual=False)

    def forward(self, x, cal=None):
        return self.inner(x)


class PBLSTMV2(nn.Module):
    """Paper-described PB-LSTM generalized to an L-step input and H-step output.

    Same layout as the L6/H1 adaptation used earlier: a 32-unit temporal dense
    layer, three bidirectional 32-unit LSTM layers per intersection, and a
    three-layer 16-unit MLP on observed hour/weekday/month and one-hot site.
    """

    def __init__(self, seq_len: int, pred_len: int, n_nodes: int, n_channels: int = 12, **_: object):
        super().__init__()
        self.seq_len, self.pred_len, self.n_nodes, self.n_channels = seq_len, pred_len, n_nodes, n_channels
        self.temporal_dense = nn.Sequential(nn.Linear(n_channels, 32), nn.ReLU())
        self.recurrent = nn.LSTM(32, 32, num_layers=3, bidirectional=True, batch_first=True)
        self.stamp = nn.Sequential(nn.Linear(3 + n_nodes, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
                                   nn.Linear(16, 16), nn.ReLU())
        self.output = nn.Linear(64 + 16, pred_len * n_channels)
        self.register_buffer("site_ids", torch.eye(n_nodes), persistent=False)

    def forward(self, x, cal):
        batch = x.shape[0]
        history = x.permute(0, 2, 1, 3).reshape(batch * self.n_nodes, self.seq_len, self.n_channels)
        _, (hidden, _) = self.recurrent(self.temporal_dense(history))
        temporal = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        last = cal[:, -1].float()
        stamp_values = torch.stack((last[:, 0] / 288.0, last[:, 1] / 6.0, last[:, 2] / 11.0), dim=-1)
        stamp_values = stamp_values[:, None, :].expand(-1, self.n_nodes, -1).reshape(-1, 3)
        site = self.site_ids.unsqueeze(0).expand(batch, -1, -1).reshape(-1, self.n_nodes)
        stamp = self.stamp(torch.cat((stamp_values, site), dim=-1))
        forecast = self.output(torch.cat((temporal, stamp), dim=-1))
        return forecast.reshape(batch, self.n_nodes, self.pred_len, self.n_channels).permute(0, 2, 1, 3)


# ----------------------------------------------------------------------------
# Node-based official baselines (each movement series is one node)
# ----------------------------------------------------------------------------

def _basicts_module(name: str):
    """Import a vendored BasicTS arch package without installing BasicTS."""
    root = VENDOR / "basicts"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "basicts.utils.misc" not in sys.modules:
        # D2STGNN imports one helper from the BasicTS package; this is the
        # same two-line function at the pinned commit.
        misc = types.ModuleType("basicts.utils.misc")

        def remove_nan_inf(tensor):
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            return torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)

        misc.remove_nan_inf = remove_nan_inf
        sys.modules.setdefault("basicts", types.ModuleType("basicts"))
        sys.modules.setdefault("basicts.utils", types.ModuleType("basicts.utils"))
        sys.modules["basicts.utils.misc"] = misc
    return importlib.import_module(f"baselines.{name}.arch")


class _NodeBaseline(nn.Module):
    """Reshape [B,L,N,C] to [B,L,N*C,F] with model-specific calendar channels."""

    calendar_mode = "none"  # none | tod | tod_dow_fraction | tod_dow_index

    def __init__(self, seq_len, pred_len, n_nodes, n_channels):
        super().__init__()
        self.seq_len, self.pred_len, self.n_nodes, self.n_channels = seq_len, pred_len, n_nodes, n_channels
        self.n_series = n_nodes * n_channels

    def node_input(self, x, cal):
        batch = x.shape[0]
        values = x.reshape(batch, self.seq_len, self.n_series, 1)
        if self.calendar_mode == "none":
            return values
        # Slot centres keep floor(fraction * size) exact for every slot.
        tod = ((cal[..., 0].float() + 0.5) / 288.0)[:, :, None, None].expand(-1, -1, self.n_series, 1)
        if self.calendar_mode == "tod":
            return torch.cat((values, tod), dim=-1)
        if self.calendar_mode == "tod_dow_fraction":
            dow = ((cal[..., 1].float() + 0.5) / 7.0)[:, :, None, None].expand(-1, -1, self.n_series, 1)
        else:
            dow = cal[..., 1].float()[:, :, None, None].expand(-1, -1, self.n_series, 1)
        return torch.cat((values, tod, dow), dim=-1)

    def node_output(self, y):
        return y.reshape(y.shape[0], self.pred_len, self.n_nodes, self.n_channels)


class STIDV2(_NodeBaseline):
    calendar_mode = "tod_dow_fraction"

    def __init__(self, seq_len, pred_len, n_nodes, n_channels=12, **_):
        super().__init__(seq_len, pred_len, n_nodes, n_channels)
        from .vendor.stid_arch import STID
        self.official = STID(num_nodes=self.n_series, input_len=seq_len, input_dim=3, embed_dim=32,
                             output_len=pred_len, num_layer=3, if_node=True, node_dim=32,
                             if_T_i_D=True, if_D_i_W=True, temp_dim_tid=32, temp_dim_diw=32,
                             time_of_day_size=288, day_of_week_size=7)

    def forward(self, x, cal):
        return self.node_output(self.official(self.node_input(x, cal), None, 0, 0, self.training))


class STAEformerV2(nn.Module):
    """Official STAEformer with intersections as nodes and 12 movement channels.

    This keeps the multivariate adaptation used in the earlier comparisons;
    treating every movement as a node would make spatial attention ~100x
    more expensive.  The pinned official ``forward`` reads time-of-day and
    day-of-week from input channels 1 and 2, which are movement counts here,
    so ``forward`` below repeats the official computation with the calendar
    taken from ``cal`` instead.  Every module and the default PEMS
    configuration are the official ones; ``test_staeformer_forward_matches_official``
    checks the equivalence on a single-channel input.

    ``movement_nodes=True`` opts into N*C spatial nodes with one value channel;
    all official modules and calendar-enabled computation remain unchanged.
    """

    def __init__(self, seq_len, pred_len, n_nodes, n_channels=12, *, official_source=None,
                 movement_nodes=False, **_):
        super().__init__()
        from .staeformer_adapter import load_official_class
        self.movement_nodes = movement_nodes
        self.n_nodes, self.n_channels = n_nodes, n_channels
        nodes, channels = (n_nodes * n_channels, 1) if movement_nodes else (n_nodes, n_channels)
        self.official = load_official_class(official_source)(
            num_nodes=nodes, in_steps=seq_len, out_steps=pred_len, steps_per_day=288,
            input_dim=channels, output_dim=channels, input_embedding_dim=24, tod_embedding_dim=24,
            dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=80,
            feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1, use_mixed_proj=True)

    def forward(self, x, cal):
        model = self.official
        if self.movement_nodes:
            x = x.reshape(x.shape[0], x.shape[1], self.n_nodes * self.n_channels, 1)
        batch, steps, nodes, _ = x.shape
        features = [model.input_proj(x)]
        tod = cal[..., 0].long()[:, :, None].expand(-1, -1, nodes)
        dow = cal[..., 1].long()[:, :, None].expand(-1, -1, nodes)
        features.append(model.tod_embedding(tod))
        features.append(model.dow_embedding(dow))
        features.append(model.adaptive_embedding.expand(size=(batch, *model.adaptive_embedding.shape)))
        h = torch.cat(features, dim=-1)
        for attn in model.attn_layers_t:
            h = attn(h, dim=1)
        for attn in model.attn_layers_s:
            h = attn(h, dim=2)
        out = h.transpose(1, 2).reshape(batch, nodes, steps * model.model_dim)
        out = model.output_proj(out).view(batch, nodes, model.out_steps, model.output_dim)
        out = out.transpose(1, 2)
        return out.reshape(batch, model.out_steps, self.n_nodes, self.n_channels) if self.movement_nodes else out


class GWNetV2(_NodeBaseline):
    calendar_mode = "tod"

    def __init__(self, seq_len, pred_len, n_nodes, n_channels=12, *, token_graph=None, **_):
        super().__init__(seq_len, pred_len, n_nodes, n_channels)
        supports = node_supports(token_graph)
        arch = _basicts_module("GWNet")
        self.official = arch.GraphWaveNet(
            num_nodes=self.n_series, dropout=0.3,
            supports=[torch.tensor(supports["forward"]), torch.tensor(supports["backward"])],
            gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=pred_len,
            residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
            kernel_size=2, blocks=4, layers=2)
        for index, support in enumerate(self.official.supports):
            self.register_buffer(f"support_{index}", support, persistent=False)

    def forward(self, x, cal):
        self.official.supports = [getattr(self, f"support_{i}") for i in range(2)]
        return self.node_output(self.official(self.node_input(x, cal), None, 0, 0, self.training))


class AGCRNV2(_NodeBaseline):
    def __init__(self, seq_len, pred_len, n_nodes, n_channels=12, **_):
        super().__init__(seq_len, pred_len, n_nodes, n_channels)
        arch = _basicts_module("AGCRN")
        self.official = arch.AGCRN(num_nodes=self.n_series, input_dim=1, rnn_units=64, output_dim=1,
                                   horizon=pred_len, num_layers=2, default_graph=True, embed_dim=10,
                                   cheb_k=2)

    def forward(self, x, cal=None):
        return self.node_output(self.official(self.node_input(x, cal), None, 0, 0, self.training))


class STGCNV2(_NodeBaseline):
    def __init__(self, seq_len, pred_len, n_nodes, n_channels=12, *, token_graph=None, **_):
        super().__init__(seq_len, pred_len, n_nodes, n_channels)
        laplacian = torch.tensor(node_supports(token_graph)["laplacian"])
        arch = _basicts_module("STGCN")
        self.official = arch.STGCN(
            Kt=3, Ks=3, blocks=[[1], [64, 16, 64], [64, 16, 64], [128, 128], [pred_len]], T=seq_len,
            n_vertex=self.n_series, act_func="glu", graph_conv_type="cheb_graph_conv",
            gso=laplacian, bias=True, droprate=0.5)

    def forward(self, x, cal=None):
        return self.node_output(self.official(self.node_input(x, cal), None, 0, 0, self.training))


class D2STGNNV2(_NodeBaseline):
    calendar_mode = "tod_dow_fraction"

    def __init__(self, seq_len, pred_len, n_nodes, n_channels=12, *, token_graph=None, **_):
        super().__init__(seq_len, pred_len, n_nodes, n_channels)
        supports = node_supports(token_graph)
        arch = _basicts_module("D2STGNN")
        self.official = arch.D2STGNN(
            num_feat=1, num_hidden=32, dropout=0.1, seq_length=pred_len, k_t=3, k_s=2, gap=3,
            num_nodes=self.n_series,
            adjs=[torch.tensor(supports["forward"]), torch.tensor(supports["backward"])],
            num_layers=5, num_modalities=2, node_hidden=12, time_emb_dim=12,
            time_in_day_size=288, day_in_week_size=7)

    def forward(self, x, cal):
        return self.node_output(self.official(self.node_input(x, cal), None, 0, 0, self.training))


from .himnet_adapter import HimNetV2


MODELS = {
    "himnet": HimNetV2,
    "st_itransformer": STiTransformerV2,
    "itransformer": ITransformerV2,
    "dlinear": DLinearV2,
    "pb_lstm": PBLSTMV2,
    "stid": STIDV2,
    "staeformer": STAEformerV2,
    "gwnet": GWNetV2,
    "agcrn": AGCRNV2,
    "stgcn": STGCNV2,
    "d2stgnn": D2STGNNV2,
}


def build(name: str, *, seq_len: int, pred_len: int, n_nodes: int, n_channels: int = 12,
          token_graph=None, official_source=None, **options) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown v2 model {name!r}")
    return MODELS[name](seq_len, pred_len, n_nodes, n_channels, token_graph=token_graph,
                        official_source=official_source, **options)
