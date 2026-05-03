# ctg_model.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
try:
    from torch_geometric.nn.models.tgn import (
        IdentityMessage,
        LastAggregator,
        TGNMemory,
        TimeEncoder,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CTGNN requires torch_geometric with the TGN modules available. "
        "Install PyTorch Geometric before running the continuous-time model."
    ) from exc


# ---------------------------------------------------------------------
# MPS-compatible TGNMemory
# ---------------------------------------------------------------------

class _Float32TGNMemory(TGNMemory):
    """Drop-in TGNMemory that keeps all timestamps as float32.

    MPS (Apple Metal) does not support scatter_reduce_ with reduce='amax' on
    int64.  PyG's TGNMemory initialises both the message-store timestamp slots
    and the last_update buffer as torch.long.  In training mode update_state
    flushes memory *before* writing new messages, so the very first call hits
    those Long tensors and causes a runtime error on MPS.

    This subclass re-initialises every timestamp to float32 at construction and
    reset time, keeping the rest of the TGN logic identical.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Re-register last_update as float32 (PyG registers it as torch.long).
        self.register_buffer('last_update', self.last_update.float())

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        t = self.memory.new_empty((0,), device=self.device, dtype=torch.float32)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        self.msg_s_store = {j: (i, i, t, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, t, msg) for j in range(self.num_nodes)}


# ---------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or 2 * dim
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ff(self.norm(x))


class StabilizedForecastHead(nn.Module):
    """
    Generic stabilized downstream head.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
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
        for block in self.blocks:
            pass
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


# ---------------------------------------------------------------------
# Explicit market structure embeddings
# ---------------------------------------------------------------------

class MarketStructureEmbeddings(nn.Module):
    """
    Embeddings for:
      - side        : bid / ask
      - level index : 1..L
      - event type  : add / cancel / execute

    Visible node ids:
      0..L-1         -> B1..BL
      L..2L-1        -> A1..AL
      2L             -> execution sink
    """

    def __init__(
        self,
        num_levels: int = 10,
        num_event_types: int = 3,
        embed_dim: int = 32,
        structure_dim: int = 64,
        dropout: float = 0.1,
        include_sink: bool = True,
    ):
        super().__init__()
        self.num_sides = 2
        self.num_levels = num_levels
        self.num_event_types = num_event_types
        self.embed_dim = embed_dim
        self.structure_dim = structure_dim
        self.include_sink = include_sink

        self.num_visible_locations = 2 * num_levels
        self.sink_location_id = self.num_visible_locations
        self.num_nodes = self.num_visible_locations + (1 if include_sink else 0)

        self.side_emb = nn.Embedding(self.num_sides, embed_dim)
        self.level_emb = nn.Embedding(num_levels, embed_dim)
        self.event_type_emb = nn.Embedding(num_event_types, embed_dim)

        self.observed_event_proj = nn.Sequential(
            nn.Linear(3 * embed_dim, structure_dim),
            nn.LayerNorm(structure_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(structure_dim, structure_dim),
        )

        self.location_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, structure_dim),
            nn.LayerNorm(structure_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(structure_dim, structure_dim),
        )

        self.sink_emb = nn.Parameter(torch.zeros(1, embed_dim))
        self.sink_proj = nn.Sequential(
            nn.Linear(embed_dim, structure_dim),
            nn.LayerNorm(structure_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(structure_dim, structure_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.side_emb.weight)
        nn.init.xavier_uniform_(self.level_emb.weight)
        nn.init.xavier_uniform_(self.event_type_emb.weight)
        nn.init.xavier_uniform_(self.sink_emb)

        for seq in [self.observed_event_proj, self.location_proj, self.sink_proj]:
            for module in seq:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def _normalize_level_index(self, level_idx: Tensor) -> Tensor:
        level_idx = level_idx.long()
        if level_idx.numel() == 0:
            return level_idx
        if int(level_idx.min().item()) >= 1:
            level_idx = level_idx - 1
        return level_idx.clamp_(0, self.num_levels - 1)

    def encode_observed_event(
        self,
        side_id: Tensor,
        level_idx: Tensor,
        event_type_id: Tensor,
    ) -> Tensor:
        level_idx = self._normalize_level_index(level_idx)
        side_vec = self.side_emb(side_id.long())
        level_vec = self.level_emb(level_idx)
        event_vec = self.event_type_emb(event_type_id.long())
        x = torch.cat([side_vec, level_vec, event_vec], dim=-1)
        return self.observed_event_proj(x)

    def all_visible_location_embeddings(self, device: Optional[torch.device] = None) -> Tensor:
        location_id = torch.arange(self.num_visible_locations, device=device, dtype=torch.long)
        side_id = location_id // self.num_levels
        level_idx = location_id % self.num_levels

        side_vec = self.side_emb(side_id)
        level_vec = self.level_emb(level_idx)
        x = torch.cat([side_vec, level_vec], dim=-1)
        return self.location_proj(x)

    def all_node_embeddings(self, device: Optional[torch.device] = None) -> Tensor:
        visible = self.all_visible_location_embeddings(device=device)
        if not self.include_sink:
            return visible
        sink = self.sink_proj(self.sink_emb.to(device))
        return torch.cat([visible, sink], dim=0)

    def encode_location(self, location_id: Tensor) -> Tensor:
        table = self.all_node_embeddings(device=location_id.device)
        return table[location_id.long()]


# ---------------------------------------------------------------------
# Raw message encoder for TGNMemory.update_state(...)
# ---------------------------------------------------------------------

class RawEventEncoder(nn.Module):
    def __init__(
        self,
        numeric_msg_dim: int,
        structure_dim: int,
        raw_msg_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.numeric_msg_dim = numeric_msg_dim
        self.structure_dim = structure_dim
        self.raw_msg_dim = raw_msg_dim

        self.net = nn.Sequential(
            nn.Linear(numeric_msg_dim + structure_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, raw_msg_dim),
        )

        self.skip = (
            nn.Identity()
            if numeric_msg_dim == raw_msg_dim
            else nn.Linear(numeric_msg_dim, raw_msg_dim)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if isinstance(self.skip, nn.Linear):
            nn.init.xavier_uniform_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def forward(self, numeric_msg: Tensor, structure_emb: Tensor) -> Tensor:
        x = torch.cat([numeric_msg, structure_emb], dim=-1)
        return self.net(x) + self.skip(numeric_msg)


# ---------------------------------------------------------------------
# Marked next-event head
# ---------------------------------------------------------------------

class MarkedIntensityHead(nn.Module):
    """
    Factorized marked-event head:
      1) time-gap intensity
      2) event-type logits
      3) location logits over visible book locations only (B1..BL, A1..AL)
    """

    def __init__(
        self,
        memory_dim: int,
        time_dim: int,
        structure_dim: int,
        num_event_types: int,
        structure_embeddings: MarketStructureEmbeddings,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.structure_embeddings = structure_embeddings
        self.num_event_types = num_event_types
        self.num_mark_locations = structure_embeddings.num_visible_locations
        self.event_embed_dim = structure_embeddings.embed_dim

        trunk_in = 2 * memory_dim + time_dim + 2 * structure_dim

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(hidden_dim, hidden_dim=2 * hidden_dim, dropout=dropout),
            ResidualMLPBlock(hidden_dim, hidden_dim=2 * hidden_dim, dropout=dropout),
        )

        self.time_gap_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.event_query = nn.Linear(hidden_dim, self.event_embed_dim)
        self.event_type_bias = nn.Parameter(torch.zeros(num_event_types))

        self.location_query = nn.Linear(hidden_dim, structure_dim)
        self.location_bias = nn.Parameter(torch.zeros(self.num_mark_locations))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.time_gap_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.event_query.weight)
        nn.init.zeros_(self.event_query.bias)
        nn.init.xavier_uniform_(self.location_query.weight)
        nn.init.zeros_(self.location_query.bias)
        nn.init.zeros_(self.event_type_bias)
        nn.init.zeros_(self.location_bias)

    def forward(
        self,
        src_mem: Tensor,
        dst_mem: Tensor,
        dt_emb: Tensor,
        src_loc_emb: Tensor,
        dst_loc_emb: Tensor,
    ) -> Dict[str, Tensor]:
        x = torch.cat([src_mem, dst_mem, dt_emb, src_loc_emb, dst_loc_emb], dim=-1)
        h = self.trunk(x)

        gap_intensity = F.softplus(self.time_gap_head(h)).squeeze(-1) + 1e-8

        event_query = self.event_query(h)
        event_table = self.structure_embeddings.event_type_emb.weight
        event_type_logits = event_query @ event_table.t() + self.event_type_bias

        location_query = self.location_query(h)
        location_table = self.structure_embeddings.all_visible_location_embeddings(device=h.device)
        location_logits = location_query @ location_table.t() + self.location_bias

        return {
            "gap_intensity": gap_intensity,
            "event_type_logits": event_type_logits,
            "location_logits": location_logits,
            "marked_context": h,
        }


# ---------------------------------------------------------------------
# Full-book masked attention readout
# ---------------------------------------------------------------------

class FullBookAttentionReadout(nn.Module):
    """
    Full-book readout over all 21 nodes:
      10 bids + 10 asks + 1 sink

    Input:
      memory_bank         : [B, N, memory_dim]
      node_struct_emb     : [B, N, structure_dim]
      node_populated_mask : [B, N] where True = populated / valid node

    Empty levels are mathematically masked from attention by converting:
      key_padding_mask = ~node_populated_mask
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        structure_dim: int,
        readout_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.structure_dim = structure_dim
        self.readout_dim = readout_dim

        self.node_proj = nn.Sequential(
            nn.Linear(memory_dim + structure_dim, readout_dim),
            nn.LayerNorm(readout_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.summary_token = nn.Parameter(torch.zeros(1, 1, readout_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=readout_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(readout_dim)
        self.post_block1 = ResidualMLPBlock(readout_dim, hidden_dim=2 * readout_dim, dropout=dropout)
        self.post_block2 = ResidualMLPBlock(readout_dim, hidden_dim=2 * readout_dim, dropout=dropout)
        self.summary_norm = nn.LayerNorm(readout_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.node_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.summary_token, mean=0.0, std=0.02)

    def forward(
        self,
        memory_bank: Tensor,
        node_struct_emb: Tensor,
        node_populated_mask: Tensor,
    ) -> Tensor:
        """
        Returns:
          global_summary: [B, readout_dim]
        """
        if memory_bank.dim() != 3:
            raise ValueError("memory_bank must have shape [B, N, D].")
        if node_struct_emb.dim() != 3:
            raise ValueError("node_struct_emb must have shape [B, N, D_s].")
        if node_populated_mask.dim() != 2:
            raise ValueError("node_populated_mask must have shape [B, N].")

        B, N, _ = memory_bank.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {N}.")

        x = torch.cat([memory_bank, node_struct_emb], dim=-1)
        node_x = self.node_proj(x)

        # Extra hardening: explicitly zero out masked node representations
        node_x = node_x.masked_fill(~node_populated_mask.unsqueeze(-1), 0.0)

        token = self.summary_token.expand(B, 1, self.readout_dim)
        seq = torch.cat([token, node_x], dim=1)

        # PyTorch MHA uses True => ignore / mask out
        key_padding_mask = torch.cat(
            [
                torch.zeros(B, 1, dtype=torch.bool, device=node_populated_mask.device),
                ~node_populated_mask,
            ],
            dim=1,
        )

        attn_out, _ = self.attn(
            seq,
            seq,
            seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        seq = self.norm1(seq + attn_out)
        seq = self.post_block1(seq)
        seq = self.post_block2(seq)

        global_summary = self.summary_norm(seq[:, 0, :])
        return global_summary


# ---------------------------------------------------------------------
# Unified CTGNN
# ---------------------------------------------------------------------

class CTGNN(nn.Module):
    """
    Unified architecture:
      - TGNMemory for asynchronous state updates
      - explicit market structure embeddings
      - factorized marked next-event head
      - full-book masked attention readout with learned summary token
      - stabilized volatility head
      - modular stabilized price-move head
    """

    def __init__(
        self,
        num_nodes: int = 21,
        numeric_msg_dim: int = 8,
        num_levels: int = 10,
        num_event_types: int = 3,
        memory_dim: int = 128,
        time_dim: int = 64,
        structure_embed_dim: int = 32,
        structure_dim: int = 64,
        raw_msg_dim: int = 64,
        msg_hidden_dim: int = 128,
        marked_hidden_dim: int = 256,
        readout_dim: int = 256,
        readout_heads: int = 4,
        volatility_out_dim: int = 3,      # e.g. log-RV over 1s / 5s / 10s
        price_move_out_dim: int = 3,      # e.g. down / flat / up
        dropout: float = 0.1,
    ):
        super().__init__()

        expected_nodes = 2 * num_levels + 1
        if num_nodes != expected_nodes:
            raise ValueError(f"num_nodes must be {expected_nodes} for {num_levels} levels + sink.")

        self.num_nodes = num_nodes
        self.num_levels = num_levels
        self.num_event_types = num_event_types
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.raw_msg_dim = raw_msg_dim

        self.structure_embeddings = MarketStructureEmbeddings(
            num_levels=num_levels,
            num_event_types=num_event_types,
            embed_dim=structure_embed_dim,
            structure_dim=structure_dim,
            dropout=dropout,
            include_sink=True,
        )

        self.raw_event_encoder = RawEventEncoder(
            numeric_msg_dim=numeric_msg_dim,
            structure_dim=structure_dim,
            raw_msg_dim=raw_msg_dim,
            hidden_dim=msg_hidden_dim,
            dropout=dropout,
        )

        self.memory = _Float32TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

        self.time_enc = TimeEncoder(time_dim)

        self.marked_intensity_head = MarkedIntensityHead(
            memory_dim=memory_dim,
            time_dim=time_dim,
            structure_dim=structure_dim,
            num_event_types=num_event_types,
            structure_embeddings=self.structure_embeddings,
            hidden_dim=marked_hidden_dim,
            dropout=dropout,
        )

        self.full_book_readout = FullBookAttentionReadout(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            structure_dim=structure_dim,
            readout_dim=readout_dim,
            num_heads=readout_heads,
            dropout=dropout,
        )

        self.volatility_head = StabilizedForecastHead(
            in_dim=readout_dim,
            out_dim=volatility_out_dim,
            hidden_dim=readout_dim,
            num_blocks=2,
            dropout=dropout,
        )

        self.price_move_head = StabilizedForecastHead(
            in_dim=readout_dim,
            out_dim=price_move_out_dim,
            hidden_dim=readout_dim,
            num_blocks=2,
            dropout=dropout,
        )

    # -------------------------------------------------------------
    # Static helper for full-book masking
    # -------------------------------------------------------------

    @staticmethod
    def build_node_populated_mask(
        bid_sizes: Tensor,
        ask_sizes: Tensor,
        sink_present: bool = True,
        eps: float = 1e-12,
    ) -> Tensor:
        """
        Args:
          bid_sizes: [B, L]
          ask_sizes: [B, L]

        Returns:
          node_populated_mask: [B, 2L+1], bool
            True  = populated / valid / should be visible to attention
            False = empty / unpopulated / should be masked out
        """
        if bid_sizes.dim() != 2 or ask_sizes.dim() != 2:
            raise ValueError("bid_sizes and ask_sizes must be [B, L].")
        if bid_sizes.shape != ask_sizes.shape:
            raise ValueError("bid_sizes and ask_sizes must have the same shape.")

        B, L = bid_sizes.shape
        bid_mask = bid_sizes > eps
        ask_mask = ask_sizes > eps
        sink_mask = torch.full((B, 1), bool(sink_present), dtype=torch.bool, device=bid_sizes.device)
        return torch.cat([bid_mask, ask_mask, sink_mask], dim=1)

    # -------------------------------------------------------------
    # Memory update path
    # -------------------------------------------------------------

    def encode_raw_message(
        self,
        numeric_msg: Tensor,
        side_id: Tensor,
        level_idx: Tensor,
        event_type_id: Tensor,
    ) -> Tensor:
        structure_emb = self.structure_embeddings.encode_observed_event(
            side_id=side_id,
            level_idx=level_idx,
            event_type_id=event_type_id,
        )
        return self.raw_event_encoder(numeric_msg, structure_emb)

    def update_memory(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        numeric_msg: Tensor,
        side_id: Tensor,
        level_idx: Tensor,
        event_type_id: Tensor,
    ) -> Tensor:
        raw_msg = self.encode_raw_message(
            numeric_msg=numeric_msg,
            side_id=side_id,
            level_idx=level_idx,
            event_type_id=event_type_id,
        )
        # PyG TGNMemory keeps pending messages internally across events. Using a
        # stop-gradient raw message keeps the asynchronous state update stable
        # under truncated BPTT while the marked-event and readout heads remain
        # fully trainable from the current-step losses.
        # MPS (Apple Metal) does not support scatter_reduce_ with reduce='amax'
        # on int64. PyG's LastAggregator calls scatter_argmax(t, ...) internally,
        # so we cast t to float32 here. Timestamp ordering is preserved exactly
        # since float32 has sufficient range for relative microsecond values over
        # any practical training window.
        self.memory.update_state(src, dst, t.float(), raw_msg.detach())
        return raw_msg

    # -------------------------------------------------------------
    # Marked next-event path
    # -------------------------------------------------------------

    def compute_marked_outputs(
        self,
        src: Tensor,
        dst: Tensor,
        dt: Tensor,
        src_location_id: Tensor,
        dst_location_id: Tensor,
    ) -> Dict[str, Tensor]:
        src_mem, _ = self.memory(src)
        dst_mem, _ = self.memory(dst)

        dt = dt.float()
        dt_emb = self.time_enc(dt)

        src_loc_emb = self.structure_embeddings.encode_location(src_location_id.long())
        dst_loc_emb = self.structure_embeddings.encode_location(dst_location_id.long())

        return self.marked_intensity_head(
            src_mem=src_mem,
            dst_mem=dst_mem,
            dt_emb=dt_emb,
            src_loc_emb=src_loc_emb,
            dst_loc_emb=dst_loc_emb,
        )

    # -------------------------------------------------------------
    # Full-book readout path
    # -------------------------------------------------------------

    def get_full_memory_bank(self) -> Tensor:
        node_ids = torch.arange(self.num_nodes, device=self.structure_embeddings.side_emb.weight.device)
        memory_bank, _ = self.memory(node_ids)   # [N, memory_dim]
        return memory_bank

    def compute_book_summary(
        self,
        node_populated_mask: Tensor,
    ) -> Tensor:
        """
        Args:
          node_populated_mask: [B, 21] or [21]
        """
        if node_populated_mask.dim() == 1:
            node_populated_mask = node_populated_mask.unsqueeze(0)

        if node_populated_mask.size(1) != self.num_nodes:
            raise ValueError(f"node_populated_mask must have shape [B, {self.num_nodes}]")

        B = node_populated_mask.size(0)
        device = node_populated_mask.device

        memory_bank = self.get_full_memory_bank().to(device)         # [N, D]
        memory_bank = memory_bank.unsqueeze(0).expand(B, -1, -1)     # [B, N, D]

        node_struct = self.structure_embeddings.all_node_embeddings(device=device)   # [N, Ds]
        node_struct = node_struct.unsqueeze(0).expand(B, -1, -1)                     # [B, N, Ds]

        return self.full_book_readout(
            memory_bank=memory_bank,
            node_struct_emb=node_struct,
            node_populated_mask=node_populated_mask.bool(),
        )

    def compute_volatility(
        self,
        node_populated_mask: Tensor,
    ) -> Tensor:
        summary = self.compute_book_summary(node_populated_mask=node_populated_mask)
        return self.volatility_head(summary)

    def compute_price_move(
        self,
        node_populated_mask: Tensor,
    ) -> Tensor:
        summary = self.compute_book_summary(node_populated_mask=node_populated_mask)
        return self.price_move_head(summary)

    # -------------------------------------------------------------
    # Unified forward
    # -------------------------------------------------------------

    def forward(
        self,
        node_populated_mask: Tensor,
        src: Optional[Tensor] = None,
        dst: Optional[Tensor] = None,
        dt: Optional[Tensor] = None,
        src_location_id: Optional[Tensor] = None,
        dst_location_id: Optional[Tensor] = None,
        compute_marked: bool = True,
        enable_price_move_head: bool = True,
    ) -> Dict[str, Optional[Tensor]]:
        """
        Unified forward.

        Args:
          node_populated_mask:
            [B, 21] or [21], used for full-book masked attention readout.

          compute_marked:
            If True, also compute the factorized marked-event outputs.
            Requires src, dst, dt, src_location_id, dst_location_id.

          enable_price_move_head:
            If False, skip the price-move head entirely.
            Use this when price-move targets are not finalized yet.

        Returns:
          dict containing:
            - book_summary
            - volatility
            - price_move (optional / None if disabled)
            - gap_intensity / event_type_logits / location_logits (if requested)
        """
        outputs: Dict[str, Optional[Tensor]] = {}

        summary = self.compute_book_summary(node_populated_mask=node_populated_mask)
        outputs["book_summary"] = summary
        outputs["volatility"] = self.volatility_head(summary)

        if enable_price_move_head:
            outputs["price_move"] = self.price_move_head(summary)
        else:
            outputs["price_move"] = None

        if compute_marked:
            required = [src, dst, dt, src_location_id, dst_location_id]
            if any(x is None for x in required):
                raise ValueError(
                    "compute_marked=True requires src, dst, dt, src_location_id, dst_location_id."
                )

            marked = self.compute_marked_outputs(
                src=src,
                dst=dst,
                dt=dt,
                src_location_id=src_location_id,
                dst_location_id=dst_location_id,
            )
            outputs.update(marked)

        return outputs
