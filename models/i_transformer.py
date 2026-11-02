"""Inverted-token Transformer baseline for graph TMF forecasting."""
from __future__ import annotations

import torch
from torch import nn


class ITransformerForecaster(nn.Module):
    """Encode each node-channel series as one token, as in iTransformer.

    The graph argument is accepted for compatibility with the shared training
    loop but is intentionally unused. Each of the ``N * C`` variates is
    represented by its complete look-back window, self-attention exchanges
    information across variates, and a shared projection produces the horizon.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_nodes: int,
        input_dim: int = 1,
        time_feature_dim: int = 0,
        input_context_mode: str = "none",
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        if time_feature_dim < 0:
            raise ValueError("time_feature_dim must be non-negative")
        self.time_feature_dim = int(time_feature_dim)
        if input_context_mode not in {"none", "raw"}:
            raise ValueError("input_context_mode must be 'none' or 'raw'")
        self.input_context_mode = str(input_context_mode)

        self.series_feature_dim = 1 + (1 if self.input_context_mode == "raw" else 0) + self.time_feature_dim
        self.series_embedding = nn.Linear(seq_len * self.series_feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.projection = nn.Linear(d_model, pred_len)

    def _prepare_series_inputs(
        self,
        normalized: torch.Tensor,
        time_features: torch.Tensor | None,
        input_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, n_nodes, channels = normalized.shape
        flow = normalized.permute(0, 2, 3, 1).reshape(
            bsz, n_nodes * channels, seq_len
        )
        features = flow.unsqueeze(-1)
        if self.input_context_mode == "raw":
            if input_context is None:
                raise ValueError("input_context is required for input_context_mode='raw'")
            if input_context.shape != normalized.shape:
                raise ValueError(
                    "input_context must match normalized input shape, "
                    f"expected {tuple(normalized.shape)}, got {tuple(input_context.shape)}"
                )
            context = input_context.to(device=normalized.device, dtype=normalized.dtype)
            context = context.permute(0, 2, 3, 1).reshape(
                bsz, n_nodes * channels, seq_len
            )
            features = torch.cat([features, context.unsqueeze(-1)], dim=-1)
        if self.time_feature_dim == 0:
            if time_features is not None:
                raise ValueError("time_features were provided to a flow-only model")
            return features.reshape(bsz, n_nodes * channels, seq_len * self.series_feature_dim)
        if time_features is None:
            raise ValueError("time_features are required for this model")
        if time_features.shape != (bsz, seq_len, self.time_feature_dim):
            raise ValueError(
                f"Expected time_features [B,{seq_len},{self.time_feature_dim}], "
                f"got {tuple(time_features.shape)}"
            )
        calendar = time_features.to(device=normalized.device, dtype=normalized.dtype)
        calendar = calendar.unsqueeze(1).expand(
            -1, n_nodes * channels, -1, -1
        )
        return torch.cat([features, calendar], dim=-1).reshape(
            bsz, n_nodes * channels, seq_len * self.series_feature_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del adj
        bsz, seq_len, n_nodes, channels = x.shape
        if seq_len != self.seq_len or n_nodes != self.n_nodes or channels != self.input_dim:
            raise ValueError(
                f"Expected input [B,{self.seq_len},{self.n_nodes},{self.input_dim}], "
                f"got {tuple(x.shape)}"
            )

        # Normalize each sample and variate over the look-back window.
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev

        series_inputs = self._prepare_series_inputs(normalized, time_features, input_context=x)
        encoded = self.encoder(self.series_embedding(series_inputs))
        forecast = self.projection(encoded)
        forecast = forecast.reshape(bsz, n_nodes, channels, self.pred_len).permute(0, 3, 1, 2)
        return forecast * stdev + means


class GraphITransformerForecaster(ITransformerForecaster):
    """iTransformer with graph-constrained refinement of variate tokens."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_nodes: int,
        input_dim: int = 1,
        time_feature_dim: int = 0,
        input_context_mode: str = "none",
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.05,
        use_spatial: bool = True,
        use_fusion: bool = True,
        spatial_source: str = "raw",
        spatial_mask_mode: str = "proximity",
        spatial_mask_seed: int = 314159,
        spatial_mask_strength: float = 1.0,
        spatial_layers: int = 1,
        spatial_residual_scale: float = 1.0,
        fusion_gate_bias: float = 0.0,
        fusion_mode: str = "convex",
        context_blend_weight: float = 0.0,
        context_mean_steps: int = 3,
        context_gate_enabled: bool = False,
        context_gate_bias: float = -1.0,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            n_nodes=n_nodes,
            input_dim=input_dim,
            time_feature_dim=time_feature_dim,
            input_context_mode=input_context_mode,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.use_spatial = use_spatial
        self.use_fusion = use_fusion
        if spatial_source not in {"raw", "temporal", "embedded"}:
            raise ValueError("spatial_source must be 'raw', 'temporal', or 'embedded'")
        if spatial_mask_mode not in {
            "proximity",
            "dense",
            "permuted",
            "soft_proximity",
        }:
            raise ValueError(
                "spatial_mask_mode must be 'proximity', 'dense', 'permuted', "
                "or 'soft_proximity'"
            )
        if spatial_mask_seed < 0:
            raise ValueError("spatial_mask_seed must be non-negative")
        if spatial_mask_strength < 0:
            raise ValueError("spatial_mask_strength must be non-negative")
        if spatial_layers <= 0:
            raise ValueError("spatial_layers must be positive")
        if spatial_residual_scale < 0:
            raise ValueError("spatial_residual_scale must be non-negative")
        self.spatial_source = str(spatial_source)
        self.spatial_mask_mode = str(spatial_mask_mode)
        self.spatial_mask_seed = int(spatial_mask_seed)
        self.spatial_mask_strength = float(spatial_mask_strength)
        self.spatial_layers = int(spatial_layers)
        self.spatial_residual_scale = float(spatial_residual_scale)
        self.fusion_gate_bias = float(fusion_gate_bias)
        if fusion_mode not in {"convex", "residual_correction", "signed_residual"}:
            raise ValueError(
                "fusion_mode must be 'convex', 'residual_correction', or 'signed_residual'"
            )
        self.fusion_mode = str(fusion_mode)
        if not 0.0 <= context_blend_weight <= 1.0:
            raise ValueError("context_blend_weight must be in [0, 1]")
        if context_mean_steps <= 0:
            raise ValueError("context_mean_steps must be positive")
        if (context_gate_enabled or context_blend_weight > 0.0) and context_mean_steps > seq_len:
            raise ValueError(
                "context_mean_steps must be in [1, seq_len] when level blending is enabled"
            )
        self.context_blend_weight = float(context_blend_weight)
        self.context_mean_steps = int(context_mean_steps)
        if context_gate_enabled and context_blend_weight != 0.0:
            raise ValueError("context_gate_enabled and context_blend_weight are mutually exclusive")
        self.context_gate_enabled = bool(context_gate_enabled)
        self.context_gate_bias = float(context_gate_bias)
        self.context_gate = nn.Linear(d_model, 1) if self.context_gate_enabled else None
        if self.context_gate is not None:
            nn.init.constant_(self.context_gate.bias, self.context_gate_bias)
        self.node_embedding = nn.Parameter(torch.randn(1, n_nodes, 1, d_model) * 0.02)
        self.channel_embedding = nn.Parameter(torch.randn(1, 1, input_dim, d_model) * 0.02)
        self.graph_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.graph_norm = nn.LayerNorm(d_model)
        self.graph_dropout = nn.Dropout(dropout)
        self.graph_attention_extra = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model,
                    n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(self.spatial_layers - 1)
            ]
        )
        self.graph_norm_extra = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.spatial_layers - 1)]
        )
        self.graph_dropout_extra = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(self.spatial_layers - 1)]
        )
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        nn.init.constant_(self.fusion_gate.bias, self.fusion_gate_bias)
        self.fusion_correction = None
        if self.fusion_mode == "residual_correction":
            self.fusion_correction = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            nn.init.zeros_(self.fusion_correction[-1].weight)
            nn.init.zeros_(self.fusion_correction[-1].bias)

    def _attention_mask(self, adj: torch.Tensor) -> torch.Tensor:
        """Return a token mask for the selected spatial attribution control."""
        graph = adj.to(dtype=torch.bool)
        if self.spatial_mask_mode == "dense":
            graph = torch.ones_like(graph, dtype=torch.bool)
        elif self.spatial_mask_mode == "permuted":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.spatial_mask_seed)
            permutation = torch.randperm(self.n_nodes, generator=generator).to(adj.device)
            graph = graph.index_select(0, permutation).index_select(1, permutation)
        graph = graph | torch.eye(self.n_nodes, device=adj.device, dtype=torch.bool)
        token_graph = graph.repeat_interleave(self.input_dim, dim=0).repeat_interleave(
            self.input_dim, dim=1
        )
        if self.spatial_mask_mode == "soft_proximity":
            return (~token_graph).to(dtype=adj.dtype) * -self.spatial_mask_strength
        return ~token_graph

    def _spatial_refine(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply masked graph attention with a backward-compatible first block."""
        h = inputs
        graph_out, _ = self.graph_attention(
            h,
            h,
            h,
            attn_mask=attention_mask,
            need_weights=False,
        )
        h = self.graph_norm(
            h + self.spatial_residual_scale * self.graph_dropout(graph_out)
        )
        for attention, norm, graph_dropout in zip(
            self.graph_attention_extra,
            self.graph_norm_extra,
            self.graph_dropout_extra,
        ):
            graph_out, _ = attention(
                h,
                h,
                h,
                attn_mask=attention_mask,
                need_weights=False,
            )
            h = norm(
                h + self.spatial_residual_scale * graph_dropout(graph_out)
            )
        return h

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, n_nodes, channels = x.shape
        if seq_len != self.seq_len or n_nodes != self.n_nodes or channels != self.input_dim:
            raise ValueError(
                f"Expected input [B,{self.seq_len},{self.n_nodes},{self.input_dim}], "
                f"got {tuple(x.shape)}"
            )

        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev

        series_inputs = self._prepare_series_inputs(normalized, time_features, input_context=x)
        tokens = self.series_embedding(series_inputs).reshape(
            bsz, n_nodes, channels, -1
        )
        tokens = tokens + self.node_embedding + self.channel_embedding
        tokens = tokens.reshape(bsz, n_nodes * channels, -1)
        temporal = self.encoder(tokens)

        if self.use_spatial:
            # raw is the legacy label used by existing recipes; retaining
            # its temporal-token behavior avoids changing historical runs.
            spatial_input = tokens if self.spatial_source == "embedded" else temporal
            spatial = self._spatial_refine(
                spatial_input,
                self._attention_mask(adj),
            )
            if self.use_fusion:
                fusion_input = torch.cat([temporal, spatial], dim=-1)
                fusion_logits = self.fusion_gate(fusion_input)
                if self.fusion_mode == "convex":
                    gate = torch.sigmoid(fusion_logits)
                    encoded = gate * spatial + (1.0 - gate) * temporal
                elif self.fusion_mode == "residual_correction":
                    if self.fusion_correction is None:
                        raise RuntimeError(
                            "residual_correction mode requires fusion_correction"
                        )
                    gate = torch.sigmoid(fusion_logits)
                    encoded = spatial + gate * self.fusion_correction(fusion_input)
                else:
                    encoded = spatial + torch.tanh(fusion_logits) * (temporal - spatial)
            else:
                encoded = spatial
        else:
            encoded = temporal

        forecast = self.projection(encoded)
        forecast = forecast.reshape(bsz, n_nodes, channels, self.pred_len).permute(0, 3, 1, 2)
        forecast = forecast * stdev + means
        if self.context_gate_enabled:
            if self.context_gate is None:
                raise RuntimeError("context_gate_enabled requires a context_gate module")
            recent_level = x[:, -self.context_mean_steps :].mean(dim=1, keepdim=True)
            gate = torch.sigmoid(self.context_gate(encoded)).reshape(bsz, n_nodes, channels, 1).permute(0, 3, 1, 2)
            forecast = (1.0 - gate) * forecast + gate * recent_level
        elif self.context_blend_weight > 0.0:
            # x is in the dataset-normalized scale; only observed history is used.
            recent_level = x[:, -self.context_mean_steps :].mean(dim=1, keepdim=True)
            forecast = (1.0 - self.context_blend_weight) * forecast + self.context_blend_weight * recent_level
        return forecast
