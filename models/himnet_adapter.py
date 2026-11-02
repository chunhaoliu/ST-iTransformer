"""Pinned official HimNet; movement nodes and explicit known future calendars.

PEMS04 (307 nodes) is nearest Core-20 (240); PEMS03 (358) is nearest
Extended-30 (360). Only their TOD/DOW widths differ (12/4 versus 8/8).
No upstream computation is rewritten. The shared trainer supplies normalized
labels and a monotonically increasing batch count only during training.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

COMMIT = "2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5"
SOURCE = Path(__file__).resolve().parents[1] / "external/himnet"
HASHES = {
    "HimNet.py": "48d9ef98d190970b9d3a5173e2734b680a638464da264c9cabb07b18aa4caa26",
    "HimNet.yaml": "75ef00e543078333d2fc3b1f3e11ad04d77b00cbf612be46d8e825965bb06f15",
    "HimNetRunner.py": "60ccfd6ded1d5efac4f6aca382196c98750f6079bb7aa1c791b502dc3a37439c",
    "SOURCE.txt": "5767228bb2e75cf44a63892281063aac6a3e35ead0b8b6a11007a8cf1ce1a3a7",
}


def load_official_class(source=SOURCE):
    source = Path(source)
    for name, expected in HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"HimNet pinned source changed: {name}")
    spec = importlib.util.spec_from_file_location("tmf_pinned_himnet", source / "HimNet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HimNet


class CheckpointDecoder(nn.Module):
    """Recompute one official decoder step, avoiding H copies of meta-weights.

    Curriculum NumPy draws live in HimNet.forward, outside this pure decoder;
    recomputation therefore never resamples teacher forcing. No math is changed.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.enabled = True

    def forward(self, *args):
        if self.enabled and self.training and torch.is_grad_enabled():
            return checkpoint(self.module, *args, use_reentrant=False)
        return self.module(*args)


class HimNetV2(nn.Module):
    requires_future_calendar = True

    def __init__(self, seq_len, pred_len, n_nodes, n_channels=12, **_):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.n_nodes, self.n_channels = n_nodes, n_channels
        self.batches_seen = 0
        self.official_config = "PEMS04" if n_nodes * n_channels <= 307 else "PEMS03"
        tod, dow = (12, 4) if self.official_config == "PEMS04" else (8, 8)
        self.official = load_official_class()(
            num_nodes=n_nodes * n_channels, input_dim=3, output_dim=1,
            out_steps=pred_len, hidden_dim=64, num_layers=1, cheb_k=2,
            ycov_dim=2, tod_embedding_dim=tod, dow_embedding_dim=dow,
            node_embedding_dim=16, st_embedding_dim=16, tf_decay_steps=4000,
            use_teacher_forcing=True)
        self.official.decoder = CheckpointDecoder(self.official.decoder)

    def _forward(self, x, cal, labels=None, batches_seen=None):
        if cal.shape[1] != self.seq_len + self.pred_len:
            raise ValueError("HimNet requires observed AND target-step calendar labels")
        batch = x.shape[0]
        nodes = self.n_nodes * self.n_channels
        # Slot centres preserve integer embedding lookup, as for other v2 adapters.
        cov = torch.stack(((cal[..., 0].to(x.dtype) + .5) / 288,
                           cal[..., 1].to(x.dtype)), dim=-1)
        cov = cov[:, :, None, :].expand(-1, -1, nodes, -1)
        inputs = torch.cat((x.reshape(batch, self.seq_len, nodes, 1),
                            cov[:, :self.seq_len]), dim=-1)
        target = None if labels is None else labels.reshape(batch, self.pred_len, nodes, 1)
        output = self.official(inputs, cov[:, self.seq_len:], target, batches_seen)
        return output.reshape(batch, self.pred_len, self.n_nodes, self.n_channels)

    def training_forward(self, x, cal, y_true):
        if not self.training:
            raise ValueError("Teacher forcing is training-only")
        output = self._forward(x, cal, y_true, self.batches_seen)
        self.batches_seen += 1
        return output

    def forward(self, x, cal):
        if self.training:
            raise ValueError("Training HimNet requires training_forward with labels")
        return self._forward(x, cal)
