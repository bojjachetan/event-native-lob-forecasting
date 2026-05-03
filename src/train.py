# train.py
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.ctg_model import CTGNN
from src.data.supervision_spine import build_supervision_mask, select_supervised_indices, supervision_report
from src.utils.device import describe_device, resolve_device
from src.utils.memory import estimate_tensor_memory_gb, log_memory
from src.utils.run_manifest import build_run_manifest, save_run_manifest
from src.utils.seeding import set_global_seed


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_FEATURE_COLS: List[str] = [
    "signed_event_size",
    "size",
    "rel_price_to_mid_bps",
    "spread_bps",
    "same_level_imbalance",
    "book_imbalance_l1",
    "node_depth_share",
    "visible_bid_depth",
    "visible_ask_depth",
]

RV_TARGET_COLS: List[str] = ["rv_1s", "rv_5s", "rv_10s"]


@dataclass
class TrainConfig:
    events_path: str
    targets_path: str
    price_move_label_col: Optional[str] = None

    num_levels: int = 10
    num_event_types: int = 3
    num_nodes: int = 21

    feature_cols: Tuple[str, ...] = tuple(DEFAULT_FEATURE_COLS)
    memory_dim: int = 128
    time_dim: int = 64
    structure_embed_dim: int = 32
    structure_dim: int = 64
    raw_msg_dim: int = 64
    msg_hidden_dim: int = 128
    marked_hidden_dim: int = 256
    readout_dim: int = 256
    readout_heads: int = 4
    dropout: float = 0.1
    volatility_out_dim: int = 3
    price_move_out_dim: int = 3

    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    epochs: int = 5
    chunk_size: int = 256
    truncated_bptt: bool = True

    mc_samples: int = 10
    mc_samples_train: Optional[int] = None
    mc_samples_eval: Optional[int] = None
    enable_price_move_head: bool = False

    replay_all_events: bool = True
    train_on_spine: bool = False
    eval_on_spine: bool = False
    supervision_mode: str = "all_events"
    supervision_interval_us: int = 250_000
    supervision_every_n: int = 10
    include_large_events: bool = False
    large_event_quantile: float = 0.95
    max_events_per_run: Optional[int] = None
    dry_run_shapes: bool = False
    progress_every_events: int = 10_000

    normalize_vol_targets: bool = False
    target_scaler: str = "standard"
    out_dir: Optional[str] = None

    w_gap_nll: float = 1.0
    w_event_type: float = 1.0
    w_location: float = 1.0
    w_volatility: float = 1.0
    w_price_move: float = 1.0

    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    set_global_seed(seed, deterministic=True)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def _bid_size_cols(num_levels: int) -> List[str]:
    return [f"bid_sz_{i}" for i in range(1, num_levels + 1)]


def _ask_size_cols(num_levels: int) -> List[str]:
    return [f"ask_sz_{i}" for i in range(1, num_levels + 1)]


def _required_event_cols(cfg: TrainConfig) -> List[str]:
    cols = [
        "event_id",
        "t_us",
        "event_type_code",
        "side_code",
        "level",
        "node_id",
    ]
    cols += list(cfg.feature_cols)
    cols += _bid_size_cols(cfg.num_levels)
    cols += _ask_size_cols(cfg.num_levels)
    return cols


def _required_target_cols(cfg: TrainConfig) -> List[str]:
    cols = ["event_id"] + RV_TARGET_COLS
    if cfg.enable_price_move_head:
        if cfg.price_move_label_col is None:
            raise ValueError(
                "enable_price_move_head=True requires price_move_label_col to be set."
            )
        cols.append(cfg.price_move_label_col)
    return cols


def load_and_merge_tables(cfg: TrainConfig) -> pd.DataFrame:
    events_df = pd.read_parquet(cfg.events_path)
    targets_df = pd.read_parquet(cfg.targets_path)

    missing_events = [c for c in _required_event_cols(cfg) if c not in events_df.columns]
    if missing_events:
        raise ValueError(f"events.parquet missing columns: {missing_events}")

    missing_targets = [c for c in _required_target_cols(cfg) if c not in targets_df.columns]
    if missing_targets:
        raise ValueError(f"targets.parquet missing columns: {missing_targets}")

    df = events_df.merge(
        targets_df[_required_target_cols(cfg)],
        on="event_id",
        how="inner",
        validate="one_to_one",
    )

    # Stable chronological sort
    df = df.reset_index(drop=False).rename(columns={"index": "_orig_order"})
    df = df.sort_values(["t_us", "_orig_order"], kind="mergesort").reset_index(drop=True)

    # Relative integer microseconds for TGNMemory
    df["t_rel_us"] = df["t_us"].astype(np.int64) - int(df["t_us"].iloc[0])

    if cfg.max_events_per_run is not None:
        df = df.iloc[: int(cfg.max_events_per_run)].reset_index(drop=True)
        df["t_rel_us"] = df["t_us"].astype(np.int64) - int(df["t_us"].iloc[0])

    return df


def dataframe_to_tensors(cfg: TrainConfig, df: pd.DataFrame) -> Dict[str, Tensor]:
    n = len(df)
    if n == 0:
        raise ValueError("Merged dataframe is empty.")

    bid_sz_cols = _bid_size_cols(cfg.num_levels)
    ask_sz_cols = _ask_size_cols(cfg.num_levels)

    numeric_msg = torch.tensor(
        df[list(cfg.feature_cols)].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    t_rel_us = torch.tensor(df["t_rel_us"].to_numpy(dtype=np.int64), dtype=torch.long)
    src = torch.tensor(df["node_id"].to_numpy(dtype=np.int64), dtype=torch.long)

    # Events interact with the execution sink node
    sink_id = 2 * cfg.num_levels
    dst = torch.full((n,), sink_id, dtype=torch.long)

    side_id = torch.tensor(df["side_code"].to_numpy(dtype=np.int64), dtype=torch.long)
    level_idx = torch.tensor(df["level"].to_numpy(dtype=np.int64), dtype=torch.long)
    event_type_id = torch.tensor(df["event_type_code"].to_numpy(dtype=np.int64), dtype=torch.long)

    bid_sizes = torch.tensor(
        df[bid_sz_cols].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    ask_sizes = torch.tensor(
        df[ask_sz_cols].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    node_populated_mask = CTGNN.build_node_populated_mask(
        bid_sizes=bid_sizes,
        ask_sizes=ask_sizes,
        sink_present=True,
    )

    vol_targets = torch.tensor(
        df[RV_TARGET_COLS].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    out: Dict[str, Tensor] = {
        "event_id": torch.tensor(df["event_id"].to_numpy(dtype=np.int64), dtype=torch.long),
        "t_us": torch.tensor(df["t_us"].to_numpy(dtype=np.int64), dtype=torch.long),
        "t_rel_us": t_rel_us,
        "src": src,
        "dst": dst,
        "side_id": side_id,
        "level_idx": level_idx,
        "event_type_id": event_type_id,
        "numeric_msg": numeric_msg,
        "node_populated_mask": node_populated_mask,
        "vol_targets": vol_targets,
    }

    if cfg.enable_price_move_head:
        price_move = torch.tensor(
            df[cfg.price_move_label_col].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )
        out["price_move_target"] = price_move

    return out


def prepare_sequence_tensors(tensors: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Build fold-local current-event labels for causal marked-process training.

    The marked head scores event `i` from memory state containing events
    strictly before `i`; only after this loss/readout is computed should event
    `i` be replayed into memory. The first event has no previous gap and is
    excluded from marked losses via `marked_valid`.
    """
    if "t_rel_us" not in tensors:
        raise KeyError("prepare_sequence_tensors requires `t_rel_us` in tensors.")

    n = int(tensors["t_rel_us"].shape[0])
    if n == 0:
        raise ValueError("Cannot prepare sequence targets for an empty tensor dict.")

    out: Dict[str, Tensor] = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in tensors.items()
    }

    out["t_rel_us"] = out["t_rel_us"] - out["t_rel_us"][0]

    marked_valid = torch.zeros(n, dtype=torch.bool)
    current_dt_sec = torch.zeros(n, dtype=torch.float32)

    if n > 1:
        marked_valid[1:] = True

        dt_us = (
            out["t_rel_us"][1:].to(torch.float32) - out["t_rel_us"][:-1].to(torch.float32)
        ).clamp_min_(1.0)
        current_dt_sec[1:] = dt_us / 1e6

    out["marked_valid"] = marked_valid
    out["current_dt_sec"] = current_dt_sec
    return out


@dataclass
class StandardTargetScaler:
    mean: List[float]
    std: List[float]
    fit_scope: str = "train_fold_only"

    @classmethod
    def fit(cls, y: Tensor) -> "StandardTargetScaler":
        mean = y.mean(dim=0)
        std = y.std(dim=0, unbiased=False).clamp_min(1e-6)
        return cls(mean=mean.cpu().tolist(), std=std.cpu().tolist())

    def transform(self, y: Tensor) -> Tensor:
        mean = torch.tensor(self.mean, dtype=y.dtype, device=y.device)
        std = torch.tensor(self.std, dtype=y.dtype, device=y.device)
        return (y - mean) / std

    def inverse_transform(self, y: Tensor | np.ndarray) -> Tensor | np.ndarray:
        if isinstance(y, np.ndarray):
            return y * np.asarray(self.std, dtype=np.float64) + np.asarray(self.mean, dtype=np.float64)
        mean = torch.tensor(self.mean, dtype=y.dtype, device=y.device)
        std = torch.tensor(self.std, dtype=y.dtype, device=y.device)
        return y * std + mean

    def to_dict(self) -> Dict[str, object]:
        return {"mean": self.mean, "std": self.std, "fit_scope": self.fit_scope, "type": "standard"}


def apply_target_scaler(tensors: Dict[str, Tensor], scaler: StandardTargetScaler) -> Dict[str, Tensor]:
    out = dict(tensors)
    out["vol_targets"] = scaler.transform(out["vol_targets"])
    return out


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------


def multi_sample_time_gap_nll(
    model: CTGNN,
    src: Tensor,
    dst: Tensor,
    src_location_id: Tensor,
    dst_location_id: Tensor,
    dt_obs_sec: Tensor,
    mc_samples: int = 10,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Time-gap NLL with K-sample Monte Carlo approximation:

      L_gap = -log λ(Δt) + ∫_0^Δt λ(s) ds
            ≈ -log λ(Δt) + Δt * (1/K) Σ_k λ(u_k),  u_k ~ Uniform(0, Δt)

    Returns:
      gap_nll, observed_marked_outputs
    """
    eps = 1e-8

    observed = model.compute_marked_outputs(
        src=src,
        dst=dst,
        dt=dt_obs_sec,
        src_location_id=src_location_id,
        dst_location_id=dst_location_id,
    )

    lambda_obs = observed["gap_intensity"]
    event_term = -torch.log(lambda_obs + eps)

    # K-sample Monte Carlo integration
    K = mc_samples
    dt_scalar = dt_obs_sec.view(-1)  # [1]
    u = torch.rand(K, device=dt_obs_sec.device, dtype=dt_obs_sec.dtype)
    dt_mc = u * dt_scalar.repeat(K)  # [K]

    src_mc = src.repeat(K)
    dst_mc = dst.repeat(K)
    src_loc_mc = src_location_id.repeat(K)
    dst_loc_mc = dst_location_id.repeat(K)

    mc_out = model.compute_marked_outputs(
        src=src_mc,
        dst=dst_mc,
        dt=dt_mc,
        src_location_id=src_loc_mc,
        dst_location_id=dst_loc_mc,
    )

    lambda_mc = mc_out["gap_intensity"]  # [K]
    integral_term = dt_scalar * lambda_mc.mean()

    gap_nll = event_term.mean() + integral_term
    return gap_nll, observed


def compute_event_losses(
    model: CTGNN,
    tensors: Dict[str, Tensor],
    idx: int,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, Tensor]:
    """
    Score the current supervised event from pre-update memory.

    This ordering is non-negotiable for temporal integrity: event `idx` is not
    written into TGNMemory until after its supervised losses have been computed.
    Non-supervised events use `replay_event_no_grad` below and never trigger the
    expensive full-book readout or marked losses.
    """
    src = tensors["src"][idx : idx + 1].to(device)
    dst = tensors["dst"][idx : idx + 1].to(device)

    node_mask = tensors["node_populated_mask"][idx : idx + 1].to(device)
    vol_target = tensors["vol_targets"][idx : idx + 1].to(device)

    downstream = model(
        node_populated_mask=node_mask,
        compute_marked=False,
        enable_price_move_head=cfg.enable_price_move_head,
    )

    volatility_mse = F.mse_loss(
        downstream["volatility"],
        vol_target,
    )

    if cfg.enable_price_move_head:
        price_move_target = tensors["price_move_target"][idx : idx + 1].to(device)
        price_move_ce = F.cross_entropy(
            downstream["price_move"],
            price_move_target,
        )
    else:
        price_move_ce = torch.zeros((), dtype=torch.float32, device=device)

    marked_valid = bool(tensors["marked_valid"][idx].item())
    if marked_valid:
        mark_src = tensors["src"][idx - 1 : idx].to(device)
        mark_dst = tensors["dst"][idx - 1 : idx].to(device)
        current_event_type_id = tensors["event_type_id"][idx : idx + 1].to(device)
        current_location_target = tensors["src"][idx : idx + 1].to(device)
        current_dt_sec = tensors["current_dt_sec"][idx : idx + 1].to(device)
        sink_location_id = torch.full_like(mark_src, 2 * cfg.num_levels)

        gap_nll, observed = multi_sample_time_gap_nll(
            model=model,
            src=mark_src,
            dst=mark_dst,
            src_location_id=mark_src,
            dst_location_id=sink_location_id,
            dt_obs_sec=current_dt_sec,
            mc_samples=cfg.mc_samples_train if cfg.mc_samples_train is not None else cfg.mc_samples,
        )

        event_type_ce = F.cross_entropy(
            observed["event_type_logits"],
            current_event_type_id,
        )
        location_ce = F.cross_entropy(
            observed["location_logits"],
            current_location_target,
        )
    else:
        gap_nll = torch.zeros((), dtype=torch.float32, device=device)
        event_type_ce = torch.zeros((), dtype=torch.float32, device=device)
        location_ce = torch.zeros((), dtype=torch.float32, device=device)

    total = (
        cfg.w_gap_nll * gap_nll
        + cfg.w_event_type * event_type_ce
        + cfg.w_location * location_ce
        + cfg.w_volatility * volatility_mse
        + cfg.w_price_move * price_move_ce
    )

    return {
        "total": total,
        "gap_nll": gap_nll.detach(),
        "event_type_ce": event_type_ce.detach(),
        "location_ce": location_ce.detach(),
        "volatility_mse": volatility_mse.detach(),
        "price_move_ce": price_move_ce.detach(),
        "marked_valid": torch.tensor(float(marked_valid), dtype=torch.float32, device=device),
    }


@torch.no_grad()
def replay_event_no_grad(
    model: CTGNN,
    tensors: Dict[str, Tensor],
    idx: int,
    device: torch.device,
) -> None:
    """Replay one observed event into memory without constructing a gradient graph."""
    model.update_memory(
        src=tensors["src"][idx : idx + 1].to(device),
        dst=tensors["dst"][idx : idx + 1].to(device),
        t=tensors["t_rel_us"][idx : idx + 1].to(device),
        numeric_msg=tensors["numeric_msg"][idx : idx + 1].to(device),
        side_id=tensors["side_id"][idx : idx + 1].to(device),
        level_idx=tensors["level_idx"][idx : idx + 1].to(device),
        event_type_id=tensors["event_type_id"][idx : idx + 1].to(device),
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def build_model(cfg: TrainConfig) -> CTGNN:
    model = CTGNN(
        num_nodes=cfg.num_nodes,
        numeric_msg_dim=len(cfg.feature_cols),
        num_levels=cfg.num_levels,
        num_event_types=cfg.num_event_types,
        memory_dim=cfg.memory_dim,
        time_dim=cfg.time_dim,
        structure_embed_dim=cfg.structure_embed_dim,
        structure_dim=cfg.structure_dim,
        raw_msg_dim=cfg.raw_msg_dim,
        msg_hidden_dim=cfg.msg_hidden_dim,
        marked_hidden_dim=cfg.marked_hidden_dim,
        readout_dim=cfg.readout_dim,
        readout_heads=cfg.readout_heads,
        volatility_out_dim=cfg.volatility_out_dim,
        price_move_out_dim=cfg.price_move_out_dim,
        dropout=cfg.dropout,
    )
    return model


def train_one_epoch(
    model: CTGNN,
    tensors: Dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
    supervised_mask: Optional[Tensor] = None,
    reset_memory: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    model.train()
    if reset_memory:
        model.memory.reset_state()

    n = tensors["t_rel_us"].shape[0]
    epoch_start = time.perf_counter()
    next_progress_event = max(int(cfg.progress_every_events), int(cfg.chunk_size), 1)
    if supervised_mask is None:
        supervised_mask = torch.ones(n, dtype=torch.bool)
    else:
        supervised_mask = supervised_mask.to(torch.bool).cpu()

    running = {
        "total": 0.0,
        "gap_nll": 0.0,
        "event_type_ce": 0.0,
        "location_ce": 0.0,
        "volatility_mse": 0.0,
        "price_move_ce": 0.0,
    }

    event_count = 0
    marked_event_count = 0.0
    supervised_event_count = 0

    for start in range(0, n, cfg.chunk_size):
        end = min(start + cfg.chunk_size, n)
        optimizer.zero_grad()
        supervised_in_chunk = int(supervised_mask[start:end].sum().item())
        if supervised_in_chunk == 0:
            for idx in range(start, end):
                replay_event_no_grad(model=model, tensors=tensors, idx=idx, device=device)
                if cfg.truncated_bptt:
                    model.memory.detach()
                event_count += 1
            continue

        for idx in range(start, end):
            is_supervised = bool(supervised_mask[idx].item())
            if not is_supervised:
                replay_event_no_grad(model=model, tensors=tensors, idx=idx, device=device)
                if cfg.truncated_bptt:
                    model.memory.detach()
                event_count += 1
                continue

            losses = compute_event_losses(
                model=model,
                tensors=tensors,
                idx=idx,
                cfg=cfg,
                device=device,
            )

            (losses["total"] / float(supervised_in_chunk)).backward()

            running["total"] += float(losses["total"].item())
            running["gap_nll"] += float(losses["gap_nll"].item())
            running["event_type_ce"] += float(losses["event_type_ce"].item())
            running["location_ce"] += float(losses["location_ce"].item())
            running["volatility_mse"] += float(losses["volatility_mse"].item())
            running["price_move_ce"] += float(losses["price_move_ce"].item())
            marked_event_count += float(losses["marked_valid"].item())
            supervised_event_count += 1
            event_count += 1

            replay_event_no_grad(model=model, tensors=tensors, idx=idx, device=device)

            # TGNMemory keeps message state across events. Detaching at the
            # event boundary gives us a stable event-level truncated BPTT path
            # instead of retaining stale graphs across the whole chunk.
            if cfg.truncated_bptt:
                model.memory.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()

        if cfg.truncated_bptt:
            model.memory.detach()

        if verbose and event_count >= next_progress_event:
            elapsed = time.perf_counter() - epoch_start
            rate = event_count / max(elapsed, 1e-9)
            print(
                f"Epoch {epoch:02d} progress | "
                f"replayed={event_count:,}/{n:,} | "
                f"supervised={supervised_event_count:,} | "
                f"elapsed={elapsed / 60:.1f} min | "
                f"rate={rate:.1f} events/s",
                flush=True,
            )
            next_progress_event += max(int(cfg.progress_every_events), int(cfg.chunk_size), 1)

    for key in ("total", "volatility_mse", "price_move_ce"):
        running[key] /= max(supervised_event_count, 1)
    for key in ("gap_nll", "event_type_ce", "location_ce"):
        running[key] /= max(marked_event_count, 1.0)
    running["replayed_events"] = float(event_count)
    running["supervised_events"] = float(supervised_event_count)

    if verbose:
        print(
            f"Epoch {epoch:02d} | "
            f"Total: {running['total']:.6f} | "
            f"GapNLL: {running['gap_nll']:.6f} | "
            f"EventCE: {running['event_type_ce']:.6f} | "
            f"LocCE: {running['location_ce']:.6f} | "
            f"VolMSE: {running['volatility_mse']:.6f} | "
            f"PriceCE: {running['price_move_ce']:.6f} | "
            f"Supervised: {supervised_event_count}/{event_count}"
            f" | Elapsed: {(time.perf_counter() - epoch_start) / 60:.1f} min",
            flush=True,
        )

    return running


def train(cfg: TrainConfig) -> None:
    start_time = time.time()
    set_seed(cfg.seed)
    resolved_device = resolve_device(cfg.device)
    device = torch.device(resolved_device)

    df = load_and_merge_tables(cfg)
    tensors = prepare_sequence_tensors(dataframe_to_tensors(cfg, df))
    scaler: Optional[StandardTargetScaler] = None
    if cfg.normalize_vol_targets:
        scaler = StandardTargetScaler.fit(tensors["vol_targets"])
        tensors = apply_target_scaler(tensors, scaler)

    if cfg.dry_run_shapes:
        shapes = {
            key: tuple(value.shape)
            for key, value in tensors.items()
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point
        }
        print(f"Estimated float tensor memory: {estimate_tensor_memory_gb(shapes):.4f} GiB")
        return

    supervised_mask = None
    if cfg.train_on_spine:
        indices = select_supervised_indices(
            df,
            mode=cfg.supervision_mode,
            interval_us=cfg.supervision_interval_us,
            include_large_events=cfg.include_large_events,
            size_quantile=cfg.large_event_quantile,
            every_n=cfg.supervision_every_n,
        )
        supervised_mask = build_supervision_mask(len(df), indices)
        report = supervision_report(len(df), indices, cfg.supervision_mode, cfg.supervision_interval_us)
        print(
            "Supervision spine: "
            f"{report['supervised_events']}/{report['original_events']} events "
            f"(compression {report['compression_ratio']:.2f}x)"
        )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    print(f"Loaded {len(df):,} events")
    print(describe_device(resolved_device))
    print(f"Numeric message dim: {len(cfg.feature_cols)}")
    print(f"MC samples K train: {cfg.mc_samples_train if cfg.mc_samples_train is not None else cfg.mc_samples}")
    print(f"Price-move head enabled: {cfg.enable_price_move_head}")
    log_memory("before training")

    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(
            model=model,
            tensors=tensors,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            epoch=epoch,
            supervised_mask=supervised_mask,
        )

    if cfg.out_dir is not None:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if scaler is not None:
            with (out / "target_scaler.json").open("w", encoding="utf-8") as f:
                json.dump(scaler.to_dict(), f, indent=2)
        manifest = build_run_manifest(
            args=None,
            config=cfg,
            repo_root=Path(__file__).resolve().parents[1],
            device=cfg.device,
            metadata={
                "num_reconstructed_events": int(len(df)),
                "elapsed_wall_clock_sec": time.time() - start_time,
                "peak_memory_gb": log_memory("after training"),
            },
        )
        save_run_manifest(out / "run_manifest.json", manifest)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CTGNN with joint marked-event + downstream losses.")
    parser.add_argument("--events", required=True, help="Path to events.parquet")
    parser.add_argument("--targets", required=True, help="Path to targets.parquet")
    parser.add_argument("--price-move-label-col", default=None, help="Optional price-move target column name")
    parser.add_argument(
        "--feature-cols",
        default=",".join(DEFAULT_FEATURE_COLS),
        help="Comma-separated event feature columns to use as numeric messages.",
    )

    parser.add_argument("--num-levels", type=int, default=10)
    parser.add_argument("--num-event-types", type=int, default=3)
    parser.add_argument("--num-nodes", type=int, default=21)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--structure-embed-dim", type=int, default=32)
    parser.add_argument("--structure-dim", type=int, default=64)
    parser.add_argument("--raw-msg-dim", type=int, default=64)
    parser.add_argument("--msg-hidden-dim", type=int, default=128)
    parser.add_argument("--marked-hidden-dim", type=int, default=256)
    parser.add_argument("--readout-dim", type=int, default=256)
    parser.add_argument("--readout-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--volatility-out-dim", type=int, default=3)
    parser.add_argument("--price-move-out-dim", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--mc-samples-train", type=int, default=None)
    parser.add_argument("--mc-samples-eval", type=int, default=None)
    parser.add_argument("--enable-price-move-head", action="store_true")
    parser.add_argument(
        "--train-on-spine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replay all events but compute supervised losses on a selected event spine.",
    )
    parser.add_argument("--eval-on-spine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--replay-all-events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--supervision-mode",
        choices=[
            "all_events",
            "every_n_events",
            "every_100ms",
            "every_250ms",
            "every_500ms",
            "last_event_per_bucket",
            "volatility_informative",
        ],
        default="all_events",
    )
    parser.add_argument("--supervision-interval-us", type=int, default=250_000)
    parser.add_argument("--supervision-every-n", type=int, default=10)
    parser.add_argument("--include-large-events", action="store_true")
    parser.add_argument("--large-event-quantile", type=float, default=0.95)
    parser.add_argument("--max-events-per-run", type=int, default=None)
    parser.add_argument("--dry-run-shapes", action="store_true")
    parser.add_argument("--progress-every-events", type=int, default=10_000)
    parser.add_argument("--normalize-vol-targets", action="store_true")
    parser.add_argument("--target-scaler", choices=["standard"], default="standard")

    parser.add_argument("--w-gap-nll", type=float, default=1.0)
    parser.add_argument("--w-event-type", type=float, default=1.0)
    parser.add_argument("--w-location", type=float, default=1.0)
    parser.add_argument("--w-volatility", type=float, default=1.0)
    parser.add_argument("--w-price-move", type=float, default=1.0)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)

    args = parser.parse_args()
    feature_cols = tuple(x.strip() for x in args.feature_cols.split(",") if x.strip())
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty.")

    return TrainConfig(
        events_path=args.events,
        targets_path=args.targets,
        price_move_label_col=args.price_move_label_col,
        num_levels=args.num_levels,
        num_event_types=args.num_event_types,
        num_nodes=args.num_nodes,
        feature_cols=feature_cols,
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        structure_embed_dim=args.structure_embed_dim,
        structure_dim=args.structure_dim,
        raw_msg_dim=args.raw_msg_dim,
        msg_hidden_dim=args.msg_hidden_dim,
        marked_hidden_dim=args.marked_hidden_dim,
        readout_dim=args.readout_dim,
        readout_heads=args.readout_heads,
        dropout=args.dropout,
        volatility_out_dim=args.volatility_out_dim,
        price_move_out_dim=args.price_move_out_dim,
        epochs=args.epochs,
        chunk_size=args.chunk_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        mc_samples=args.mc_samples,
        mc_samples_train=args.mc_samples_train,
        mc_samples_eval=args.mc_samples_eval,
        enable_price_move_head=args.enable_price_move_head,
        replay_all_events=args.replay_all_events,
        train_on_spine=args.train_on_spine,
        eval_on_spine=args.eval_on_spine,
        supervision_mode=args.supervision_mode,
        supervision_interval_us=args.supervision_interval_us,
        supervision_every_n=args.supervision_every_n,
        include_large_events=args.include_large_events,
        large_event_quantile=args.large_event_quantile,
        max_events_per_run=args.max_events_per_run,
        dry_run_shapes=args.dry_run_shapes,
        progress_every_events=args.progress_every_events,
        normalize_vol_targets=args.normalize_vol_targets,
        target_scaler=args.target_scaler,
        w_gap_nll=args.w_gap_nll,
        w_event_type=args.w_event_type,
        w_location=args.w_location,
        w_volatility=args.w_volatility,
        w_price_move=args.w_price_move,
        device=args.device,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
