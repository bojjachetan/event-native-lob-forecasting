
"""
baselines.py

Discrete-time baselines for the continuous-time LOB paper.

Implements:
  1) DeepLOBBaseline: CNN + Inception + LSTM for 100ms snapshot sequences.
  2) StaticGCNBaseline: GCN over 1-second LOB graph snapshots.

Both models output shape [B, 3] corresponding to:
  [rv_1s, rv_5s, rv_10s]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception:  # pragma: no cover
    GCNConv = None
    global_mean_pool = None


RV_OUT_DIM = 3


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or 2 * dim
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ff(self.norm(x))


class StabilizedRegressionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = RV_OUT_DIM,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, hidden_dim=2 * hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.input_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InceptionLOBBlock(nn.Module):
    """
    DeepLOB-style multi-scale temporal filters over a 2D feature map.
    Input shape: [B, C, T, F]
    Output shape: [B, C_out, T, 1]
    """
    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.branch_1 = nn.Sequential(
            ConvBlock2d(in_channels, branch_channels, kernel_size=(1, 1), padding=(0, 0), dropout=dropout),
            ConvBlock2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0), dropout=dropout),
        )
        self.branch_2 = nn.Sequential(
            ConvBlock2d(in_channels, branch_channels, kernel_size=(1, 1), padding=(0, 0), dropout=dropout),
            ConvBlock2d(branch_channels, branch_channels, kernel_size=(5, 1), padding=(2, 0), dropout=dropout),
        )
        self.branch_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            ConvBlock2d(in_channels, branch_channels, kernel_size=(1, 1), padding=(0, 0), dropout=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat([x1, x2, x3], dim=1)


class DeepLOBBaseline(nn.Module):
    """
    DeepLOB-style baseline adapted for 3D RV regression.

    Expected input:
      x: [B, T, F]
         where T is the number of 100ms snapshots in the rolling window and
         F is the flattened LOB snapshot feature dimension.
    """
    def __init__(
        self,
        input_dim: int = 40,
        conv_channels: int = 32,
        inception_branch_channels: int = 16,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        regression_hidden_dim: int = 128,
        dropout: float = 0.1,
        out_dim: int = RV_OUT_DIM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.feature_extractor = nn.Sequential(
            ConvBlock2d(1, conv_channels, kernel_size=(1, 2), stride=(1, 2), padding=(0, 0), dropout=dropout / 2),
            ConvBlock2d(conv_channels, conv_channels, kernel_size=(4, 1), padding=(2, 0), dropout=dropout / 2),
            ConvBlock2d(conv_channels, conv_channels, kernel_size=(4, 1), padding=(2, 0), dropout=dropout / 2),
        )

        self.feature_compressor = nn.Sequential(
            ConvBlock2d(conv_channels, conv_channels, kernel_size=(1, max(1, input_dim // 2)), padding=(0, 0), dropout=dropout / 2),
        )

        self.inception = InceptionLOBBlock(
            in_channels=conv_channels,
            branch_channels=inception_branch_channels,
            dropout=dropout / 2,
        )

        inception_out_channels = 3 * inception_branch_channels

        self.lstm = nn.LSTM(
            input_size=inception_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = StabilizedRegressionHead(
            in_dim=lstm_hidden_dim,
            out_dim=out_dim,
            hidden_dim=regression_hidden_dim,
            num_blocks=2,
            dropout=dropout,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"DeepLOBBaseline expects [B, T, F], got {tuple(x.shape)}")

        _, _, feat_dim = x.shape
        if feat_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {feat_dim}")

        x = x.unsqueeze(1)                   # [B, 1, T, F]
        x = self.feature_extractor(x)        # [B, C, T', F']
        x = self.feature_compressor(x)       # [B, C, T', 1]
        x = self.inception(x)                # [B, C_inc, T', 1]
        x = x.squeeze(-1).transpose(1, 2)    # [B, T', C_inc]
        x, _ = self.lstm(x)                  # [B, T', H]
        x = x[:, -1, :]                      # [B, H]
        return self.head(x)


@dataclass
class LOBGraphSpec:
    edge_index: Tensor
    edge_weight: Optional[Tensor] = None


def build_static_lob_edge_index(
    num_levels: int = 10,
    include_self_loops: bool = False,
    device: Optional[torch.device] = None,
) -> Tensor:
    src = []
    dst = []
    ask_offset = num_levels

    for i in range(num_levels - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])

    for i in range(num_levels - 1):
        u = ask_offset + i
        v = ask_offset + i + 1
        src.extend([u, v])
        dst.extend([v, u])

    for i in range(num_levels):
        b = i
        a = ask_offset + i
        src.extend([b, a])
        dst.extend([a, b])

    if include_self_loops:
        for i in range(2 * num_levels):
            src.append(i)
            dst.append(i)

    return torch.tensor([src, dst], dtype=torch.long, device=device)


class StaticGCNBaseline(nn.Module):
    """
    Static graph baseline over 1-second LOB snapshots.

    Expected input:
      node_x: [B, N, F_node]
    """
    def __init__(
        self,
        node_feat_dim: int = 5,
        num_levels: int = 10,
        hidden_dim: int = 64,
        num_gcn_layers: int = 3,
        graph_dropout: float = 0.1,
        regression_hidden_dim: int = 128,
        out_dim: int = RV_OUT_DIM,
    ):
        super().__init__()
        if GCNConv is None or global_mean_pool is None:
            raise ImportError(
                "StaticGCNBaseline requires torch_geometric. "
                "Install PyG to use this model."
            )

        self.node_feat_dim = node_feat_dim
        self.num_levels = num_levels
        self.num_nodes = 2 * num_levels
        self.out_dim = out_dim
        self.graph_dropout = graph_dropout

        self.input_proj = nn.Sequential(
            nn.LayerNorm(node_feat_dim),
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
        )

        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)]
        )

        self.post_gcn_norm = nn.LayerNorm(hidden_dim)

        self.head = StabilizedRegressionHead(
            in_dim=hidden_dim,
            out_dim=out_dim,
            hidden_dim=regression_hidden_dim,
            num_blocks=2,
            dropout=graph_dropout,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.input_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        node_x: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if node_x.dim() != 3:
            raise ValueError(f"StaticGCNBaseline expects [B, N, F_node], got {tuple(node_x.shape)}")

        bsz, num_nodes, feat_dim = node_x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")
        if feat_dim != self.node_feat_dim:
            raise ValueError(f"Expected node_feat_dim={self.node_feat_dim}, got {feat_dim}")

        device = node_x.device
        if edge_index is None:
            edge_index = build_static_lob_edge_index(num_levels=self.num_levels, device=device)

        x = node_x.reshape(bsz * num_nodes, feat_dim)
        x = self.input_proj(x)

        base_edge_index = edge_index
        E = base_edge_index.size(1)

        batch_offsets = (
            torch.arange(bsz, device=device, dtype=torch.long)
            .repeat_interleave(E)
            * num_nodes
        )

        edge_index_batched = base_edge_index.repeat(1, bsz)
        edge_index_batched = edge_index_batched + batch_offsets.unsqueeze(0)

        edge_weight_batched = edge_weight.repeat(bsz) if edge_weight is not None else None
        if edge_weight_batched is not None:
            edge_weight_batched = edge_weight_batched.to(device)

        for conv in self.gcn_layers:
            residual = x
            x = conv(x, edge_index_batched, edge_weight=edge_weight_batched)
            x = F.gelu(x)
            x = F.dropout(x, p=self.graph_dropout, training=self.training)
            x = x + residual

        x = self.post_gcn_norm(x)
        batch_index = torch.arange(bsz, device=device, dtype=torch.long).repeat_interleave(num_nodes)
        graph_emb = global_mean_pool(x, batch_index)
        return self.head(graph_emb)


def build_baseline(name: str, **kwargs) -> nn.Module:
    name = name.lower().strip()
    if name == "deeplob":
        return DeepLOBBaseline(**kwargs)
    if name in {"static_gcn", "gcn"}:
        return StaticGCNBaseline(**kwargs)
    raise ValueError(f"Unknown baseline name: {name}")


__all__ = [
    "RV_OUT_DIM",
    "LOBGraphSpec",
    "DeepLOBBaseline",
    "StaticGCNBaseline",
    "build_static_lob_edge_index",
    "build_baseline",
]
