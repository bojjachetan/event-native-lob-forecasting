# eval_runner.py
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.make_splits import EMBARGO_US, PurgedWalkForwardFold, generate_purged_walk_forward_splits
from src.train import (
    DEFAULT_FEATURE_COLS,
    StandardTargetScaler,
    TrainConfig,
    apply_target_scaler,
    build_model,
    dataframe_to_tensors,
    load_and_merge_tables,
    multi_sample_time_gap_nll,
    prepare_sequence_tensors,
    replay_event_no_grad,
    train_one_epoch,
)
from src.data.supervision_spine import (
    build_supervision_mask,
    save_supervision_artifacts,
    select_supervised_indices,
    supervision_report,
)
from src.utils.device import describe_device, resolve_device
from src.utils.logging import format_mean_std
from src.utils.memory import estimate_tensor_memory_gb, log_memory
from src.utils.run_manifest import build_run_manifest, save_run_manifest
from src.utils.seeding import set_global_seed


RV_TARGET_COLS = ["rv_1s", "rv_5s", "rv_10s"]
HORIZONS = ("1s", "5s", "10s")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class EvalConfig:
    events_path: str
    targets_path: str
    price_move_label_col: Optional[str]
    feature_cols: Tuple[str, ...]

    train_window_us: int
    test_window_us: int
    embargo_us: int = EMBARGO_US
    step_us: Optional[int] = None
    anchored: bool = False
    min_train_events: int = 1_000
    min_test_events: int = 1_000

    seeds: Tuple[int, ...] = (42, 43, 44)
    use_log_rv_targets: bool = True
    log_rv_eps: float = 1e-8

    epochs: int = 5
    chunk_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    mc_samples: int = 10
    mc_samples_train: Optional[int] = None
    mc_samples_eval: Optional[int] = None
    enable_price_move_head: bool = False

    normalize_vol_targets: bool = False
    target_scaler: str = "standard"
    early_stopping: bool = False
    val_fraction: float = 0.2
    patience: int = 2
    min_delta: float = 1.0e-4
    early_stopping_metric: str = "composite"
    lambda_rank: float = 0.25

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

    w_gap_nll: float = 1.0
    w_event_type: float = 1.0
    w_location: float = 1.0
    w_volatility: float = 1.0
    w_price_move: float = 1.0

    num_levels: int = 10
    num_event_types: int = 3
    num_nodes: int = 21
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

    device: str = "cpu"

    out_dir: Optional[str] = None


# ---------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    set_global_seed(seed, deterministic=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_train_config(cfg: EvalConfig) -> TrainConfig:
    return TrainConfig(
        events_path=cfg.events_path,
        targets_path=cfg.targets_path,
        price_move_label_col=cfg.price_move_label_col,
        feature_cols=cfg.feature_cols,
        num_levels=cfg.num_levels,
        num_event_types=cfg.num_event_types,
        num_nodes=cfg.num_nodes,
        memory_dim=cfg.memory_dim,
        time_dim=cfg.time_dim,
        structure_embed_dim=cfg.structure_embed_dim,
        structure_dim=cfg.structure_dim,
        raw_msg_dim=cfg.raw_msg_dim,
        msg_hidden_dim=cfg.msg_hidden_dim,
        marked_hidden_dim=cfg.marked_hidden_dim,
        readout_dim=cfg.readout_dim,
        readout_heads=cfg.readout_heads,
        dropout=cfg.dropout,
        volatility_out_dim=cfg.volatility_out_dim,
        price_move_out_dim=cfg.price_move_out_dim,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip_norm=cfg.grad_clip_norm,
        epochs=cfg.epochs,
        chunk_size=cfg.chunk_size,
        mc_samples=cfg.mc_samples,
        mc_samples_train=cfg.mc_samples_train,
        mc_samples_eval=cfg.mc_samples_eval,
        enable_price_move_head=cfg.enable_price_move_head,
        replay_all_events=cfg.replay_all_events,
        train_on_spine=cfg.train_on_spine,
        eval_on_spine=cfg.eval_on_spine,
        supervision_mode=cfg.supervision_mode,
        supervision_interval_us=cfg.supervision_interval_us,
        supervision_every_n=cfg.supervision_every_n,
        include_large_events=cfg.include_large_events,
        large_event_quantile=cfg.large_event_quantile,
        max_events_per_run=cfg.max_events_per_run,
        dry_run_shapes=cfg.dry_run_shapes,
        progress_every_events=cfg.progress_every_events,
        normalize_vol_targets=cfg.normalize_vol_targets,
        target_scaler=cfg.target_scaler,
        w_gap_nll=cfg.w_gap_nll,
        w_event_type=cfg.w_event_type,
        w_location=cfg.w_location,
        w_volatility=cfg.w_volatility,
        w_price_move=cfg.w_price_move,
        device=cfg.device,
        seed=42,
    )


def subset_tensors(tensors: Dict[str, torch.Tensor], indices: np.ndarray) -> Dict[str, torch.Tensor]:
    idx = torch.as_tensor(indices, dtype=torch.long)
    out: Dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def apply_log_rv_targets(
    tensors: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    out = dict(tensors)
    out["vol_targets"] = torch.log(torch.clamp(out["vol_targets"], min=eps))
    return out


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    val = spearmanr(x, y).correlation
    if val is None or np.isnan(val):
        return 0.0
    return float(val)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def macro_f1_score_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: Optional[int] = None) -> float:
    if y_true.size == 0:
        return 0.0

    if num_classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.arange(num_classes)

    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        if tp == 0 and fp == 0 and fn == 0:
            continue

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)

    if not f1s:
        return 0.0
    return float(np.mean(f1s))


def chronological_train_val_split_indices(
    tensors: Dict[str, torch.Tensor],
    val_fraction: float,
    purge_us: int,
    min_val_events: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(tensors["t_us"].shape[0])
    if n < max(min_val_events * 2, 4):
        return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
    val_n = max(int(round(n * float(val_fraction))), min_val_events)
    val_n = min(val_n, n - 1)
    val_start = n - val_n
    val_start_t = int(tensors["t_us"][val_start].item())
    train_t = tensors["t_us"].cpu().numpy().astype(np.int64)
    train_end = int(np.searchsorted(train_t, val_start_t - int(purge_us), side="left"))
    if train_end <= 0:
        train_end = max(val_start - min_val_events, 1)
    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(val_start, n, dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) < min_val_events:
        return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
    return train_idx, val_idx


def validation_score(
    model,
    train_inner: Dict[str, torch.Tensor],
    val_inner: Dict[str, torch.Tensor],
    cfg: EvalConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    target_scaler: Optional[StandardTargetScaler],
    supervised_mask: Optional[torch.Tensor],
) -> Tuple[float, Dict[str, float]]:
    replay_sequence_into_memory(model=model, tensors=train_inner, device=device)
    fold_output = evaluate_sequence(
        model=model,
        tensors=val_inner,
        cfg=cfg,
        device=device,
        supervised_mask=supervised_mask,
        target_scaler=target_scaler,
    )
    summary = summarize_seed_results([fold_output], cfg)
    val_loss = 0.0
    for h in HORIZONS:
        val_loss += summary[f"log_rv_rmse_{h}"]
    val_loss /= 3.0
    metric = cfg.early_stopping_metric
    if metric == "val_spearman_10s":
        score = -summary["log_rv_spearman_10s"]
    elif metric == "composite":
        score = val_loss - float(cfg.lambda_rank) * summary["log_rv_spearman_10s"]
    else:
        score = val_loss
    return float(score), summary


# ---------------------------------------------------------------------
# Memory replay
# ---------------------------------------------------------------------

@torch.no_grad()
def replay_sequence_into_memory(
    model,
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """
    Rebuild consistent memory state under the final trained weights by replaying
    the train sequence once, chronologically, with no gradients.
    """
    model.eval()
    model.memory.reset_state()

    n = int(tensors["t_rel_us"].shape[0])

    for idx in range(n):
        replay_event_no_grad(model=model, tensors=tensors, idx=idx, device=device)

    model.memory.detach()


# ---------------------------------------------------------------------
# Evaluation on one fold
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_sequence(
    model,
    tensors: Dict[str, torch.Tensor],
    cfg: EvalConfig,
    device: torch.device,
    supervised_mask: Optional[torch.Tensor] = None,
    target_scaler: Optional[StandardTargetScaler] = None,
) -> Dict[str, object]:
    model.eval()

    n = int(tensors["t_rel_us"].shape[0])
    if supervised_mask is None:
        supervised_mask = torch.ones(n, dtype=torch.bool)
    else:
        supervised_mask = supervised_mask.to(torch.bool).cpu()

    event_nll_sum = 0.0
    gap_nll_sum = 0.0
    next_event_type_correct = 0
    next_location_correct = 0
    num_marked_events = 0
    num_events = int(supervised_mask.sum().item())

    vol_pred_all = []
    vol_true_all = []
    event_id_all = []
    t_us_all = []

    price_pred_all = []
    price_true_all = []

    sink_location_id_scalar = 2 * cfg.num_levels

    for idx in range(n):
        src = tensors["src"][idx : idx + 1].to(device)
        dst = tensors["dst"][idx : idx + 1].to(device)
        t_i = tensors["t_rel_us"][idx : idx + 1].to(device)

        numeric_msg = tensors["numeric_msg"][idx : idx + 1].to(device)
        side_id = tensors["side_id"][idx : idx + 1].to(device)
        level_idx = tensors["level_idx"][idx : idx + 1].to(device)
        event_type_id = tensors["event_type_id"][idx : idx + 1].to(device)
        node_mask = tensors["node_populated_mask"][idx : idx + 1].to(device)
        vol_target = tensors["vol_targets"][idx : idx + 1].to(device)
        event_id = int(tensors["event_id"][idx].item())
        event_t_us = int(tensors["t_us"][idx].item())

        should_score = bool(supervised_mask[idx].item())

        if should_score:
            downstream = model(
                node_populated_mask=node_mask,
                compute_marked=False,
                enable_price_move_head=cfg.enable_price_move_head,
            )

            vol_pred = downstream["volatility"]
            vol_true = vol_target
            if target_scaler is not None:
                vol_pred = target_scaler.inverse_transform(vol_pred)  # type: ignore[assignment]
                vol_true = target_scaler.inverse_transform(vol_true)  # type: ignore[assignment]
            vol_pred_all.append(vol_pred.cpu().numpy()[0])
            vol_true_all.append(vol_true.cpu().numpy()[0])
            event_id_all.append(event_id)
            t_us_all.append(event_t_us)

        if should_score and bool(tensors["marked_valid"][idx].item()):
            mark_src = tensors["src"][idx - 1 : idx].to(device)
            mark_dst = tensors["dst"][idx - 1 : idx].to(device)
            current_event_type_id = tensors["event_type_id"][idx : idx + 1].to(device)
            current_location_target = tensors["src"][idx : idx + 1].to(device)
            current_dt_sec = tensors["current_dt_sec"][idx : idx + 1].to(device)
            sink_location_id = torch.full_like(mark_src, sink_location_id_scalar)

            gap_nll, observed = multi_sample_time_gap_nll(
                model=model,
                src=mark_src,
                dst=mark_dst,
                src_location_id=mark_src,
                dst_location_id=sink_location_id,
                dt_obs_sec=current_dt_sec,
                mc_samples=cfg.mc_samples_eval if cfg.mc_samples_eval is not None else cfg.mc_samples,
            )

            event_type_ce = F.cross_entropy(observed["event_type_logits"], current_event_type_id)
            location_ce = F.cross_entropy(observed["location_logits"], current_location_target)
            event_nll = gap_nll + event_type_ce + location_ce

            pred_event_type = int(torch.argmax(observed["event_type_logits"], dim=-1).item())
            pred_location = int(torch.argmax(observed["location_logits"], dim=-1).item())

            next_event_type_correct += int(pred_event_type == int(current_event_type_id.item()))
            next_location_correct += int(pred_location == int(current_location_target.item()))
            event_nll_sum += float(event_nll.item())
            gap_nll_sum += float(gap_nll.item())
            num_marked_events += 1

        if should_score and cfg.enable_price_move_head:
            price_target = tensors["price_move_target"][idx : idx + 1].to(device)
            pred_price = int(torch.argmax(downstream["price_move"], dim=-1).item())
            price_pred_all.append(pred_price)
            price_true_all.append(int(price_target.item()))

        replay_event_no_grad(model=model, tensors=tensors, idx=idx, device=device)

    vol_pred_all = np.asarray(vol_pred_all, dtype=np.float64).reshape(-1, 3)
    vol_true_all = np.asarray(vol_true_all, dtype=np.float64).reshape(-1, 3)

    out: Dict[str, object] = {
        "num_events": num_events,
        "num_marked_events": num_marked_events,
        "event_nll_sum": event_nll_sum,
        "gap_nll_sum": gap_nll_sum,
        "next_event_type_correct": next_event_type_correct,
        "next_location_correct": next_location_correct,
        "vol_pred": vol_pred_all,
        "vol_true": vol_true_all,
        "event_id": np.asarray(event_id_all, dtype=np.int64),
        "t_us": np.asarray(t_us_all, dtype=np.int64),
    }

    if cfg.enable_price_move_head:
        out["price_pred"] = np.asarray(price_pred_all, dtype=np.int64)
        out["price_true"] = np.asarray(price_true_all, dtype=np.int64)

    return out


# ---------------------------------------------------------------------
# Seed-level aggregation
# ---------------------------------------------------------------------

def summarize_seed_results(
    fold_outputs: List[Dict[str, object]],
    cfg: EvalConfig,
) -> Dict[str, float]:
    total_events = int(sum(x["num_events"] for x in fold_outputs))
    if total_events == 0:
        raise ValueError("No evaluated events across folds.")

    total_marked_events = int(sum(x["num_marked_events"] for x in fold_outputs))
    if total_marked_events > 0:
        next_event_nll = sum(float(x["event_nll_sum"]) for x in fold_outputs) / total_marked_events
        next_event_type_acc = (
            sum(int(x["next_event_type_correct"]) for x in fold_outputs) / total_marked_events
        )
        next_location_acc = (
            sum(int(x["next_location_correct"]) for x in fold_outputs) / total_marked_events
        )
    else:
        next_event_nll = np.nan
        next_event_type_acc = np.nan
        next_location_acc = np.nan

    vol_pred = np.concatenate([x["vol_pred"] for x in fold_outputs], axis=0)
    vol_true = np.concatenate([x["vol_true"] for x in fold_outputs], axis=0)

    summary: Dict[str, float] = {
        "next_event_nll": float(next_event_nll),
        "next_event_type_accuracy": float(next_event_type_acc),
        "next_location_accuracy": float(next_location_acc),
        "num_marked_events": float(total_marked_events),
    }

    horizon_names = ["1s", "5s", "10s"]
    for h_idx, h_name in enumerate(horizon_names):
        y_true = vol_true[:, h_idx]
        y_pred = vol_pred[:, h_idx]

        summary[f"log_rv_rmse_{h_name}"] = rmse(y_true, y_pred)
        summary[f"log_rv_mae_{h_name}"] = mae(y_true, y_pred)
        summary[f"log_rv_spearman_{h_name}"] = safe_spearman(y_pred, y_true)

    if cfg.enable_price_move_head:
        price_pred = np.concatenate([x["price_pred"] for x in fold_outputs], axis=0)
        price_true = np.concatenate([x["price_true"] for x in fold_outputs], axis=0)

        summary["price_move_accuracy"] = accuracy_score_np(price_true, price_pred)
        summary["price_move_f1_macro"] = macro_f1_score_np(price_true, price_pred)
    else:
        summary["price_move_accuracy"] = np.nan
        summary["price_move_f1_macro"] = np.nan

    return summary


def summarize_across_seeds(seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in seed_df.columns if c != "seed"]
    rows = []

    for col in metric_cols:
        vals = seed_df[col].dropna().to_numpy(dtype=np.float64)
        if vals.size == 0:
            rows.append(
                {
                    "metric": col,
                    "mean": np.nan,
                    "std": np.nan,
                    "formatted": "",
                }
            )
            continue

        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        rows.append(
            {
                "metric": col,
                "mean": mean_val,
                "std": std_val,
                "formatted": format_mean_std(mean_val, std_val),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------

def run_purged_walk_forward_experiment(cfg: EvalConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_time = time.time()
    resolved_device = resolve_device(cfg.device)
    device = torch.device(resolved_device)
    train_cfg = make_train_config(cfg)

    df = load_and_merge_tables(train_cfg)
    full_tensors = dataframe_to_tensors(train_cfg, df)

    if cfg.use_log_rv_targets:
        full_tensors = apply_log_rv_targets(full_tensors, eps=cfg.log_rv_eps)

    if cfg.dry_run_shapes:
        shapes = {
            key: tuple(value.shape)
            for key, value in full_tensors.items()
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point
        }
        print(f"Estimated float tensor memory: {estimate_tensor_memory_gb(shapes):.4f} GiB")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    t_us = df["t_us"].to_numpy(dtype=np.int64)

    folds = list(
        generate_purged_walk_forward_splits(
            t_us=t_us,
            train_window_us=cfg.train_window_us,
            test_window_us=cfg.test_window_us,
            embargo_us=cfg.embargo_us,
            step_us=cfg.step_us,
            anchored=cfg.anchored,
            min_train_events=cfg.min_train_events,
            min_test_events=cfg.min_test_events,
        )
    )

    if not folds:
        raise ValueError("No valid purged walk-forward folds were generated.")

    print(describe_device(resolved_device))
    print(f"Purged folds: {len(folds)}")
    print(
        "Supervision: "
        f"train_on_spine={cfg.train_on_spine}, eval_on_spine={cfg.eval_on_spine}, "
        f"mode={cfg.supervision_mode}, replay_all_events={cfg.replay_all_events}"
    )
    print(
        "Workload estimate: "
        f"train_events_per_epoch_per_seed={sum(f.num_train for f in folds):,}, "
        f"total_full_supervised_train_events={sum(f.num_train for f in folds) * len(cfg.seeds) * cfg.epochs:,}, "
        f"seeds={len(cfg.seeds)}, epochs={cfg.epochs}",
        flush=True,
    )
    log_memory("after tensor load")

    prediction_dir: Optional[str] = None
    scaler_dir: Optional[str] = None
    supervision_dir: Optional[str] = None
    if cfg.out_dir is not None:
        os.makedirs(cfg.out_dir, exist_ok=True)
        prediction_dir = os.path.join(cfg.out_dir, "predictions")
        os.makedirs(prediction_dir, exist_ok=True)
        scaler_dir = os.path.join(cfg.out_dir, "scalers")
        supervision_dir = os.path.join(cfg.out_dir, "supervision")
        os.makedirs(scaler_dir, exist_ok=True)
        os.makedirs(supervision_dir, exist_ok=True)

    fold_rows = []
    seed_rows = []

    for seed in cfg.seeds:
        set_seed(seed)

        seed_fold_outputs: List[Dict[str, object]] = []

        for fold in folds:
            print(
                f"\n=== CTGNN seed={seed} fold={fold.fold_id}/{len(folds)} | "
                f"train_events={fold.num_train:,} test_events={fold.num_test:,} ===",
                flush=True,
            )
            train_tensors_raw = prepare_sequence_tensors(subset_tensors(full_tensors, fold.train_indices))
            test_tensors_raw = prepare_sequence_tensors(subset_tensors(full_tensors, fold.test_indices))
            train_df = df.iloc[fold.train_indices].reset_index(drop=True)
            test_df = df.iloc[fold.test_indices].reset_index(drop=True)

            target_scaler: Optional[StandardTargetScaler] = None
            if cfg.normalize_vol_targets:
                target_scaler = StandardTargetScaler.fit(train_tensors_raw["vol_targets"])
                train_tensors = apply_target_scaler(train_tensors_raw, target_scaler)
                test_tensors = apply_target_scaler(test_tensors_raw, target_scaler)
                if scaler_dir is not None:
                    scaler_path = Path(scaler_dir) / f"target_scaler_seed_{seed}_fold_{fold.fold_id:03d}.json"
                    scaler_payload = target_scaler.to_dict()
                    scaler_payload.update(
                        {
                            "seed": int(seed),
                            "fold_id": int(fold.fold_id),
                            "fit_event_count": int(train_tensors_raw["vol_targets"].shape[0]),
                            "target_columns": list(RV_TARGET_COLS),
                            "train_start_us": int(train_tensors_raw["t_us"][0].item()),
                            "train_end_us": int(train_tensors_raw["t_us"][-1].item()),
                        }
                    )
                    with scaler_path.open("w", encoding="utf-8") as f:
                        json.dump(scaler_payload, f, indent=2)
            else:
                train_tensors = train_tensors_raw
                test_tensors = test_tensors_raw

            train_supervised_mask: Optional[torch.Tensor] = None
            if cfg.train_on_spine:
                indices = select_supervised_indices(
                    train_df,
                    mode=cfg.supervision_mode,
                    interval_us=cfg.supervision_interval_us,
                    include_large_events=cfg.include_large_events,
                    size_quantile=cfg.large_event_quantile,
                    every_n=cfg.supervision_every_n,
                )
                train_supervised_mask = build_supervision_mask(len(train_df), indices)
                if supervision_dir is not None:
                    save_supervision_artifacts(
                        supervision_dir,
                        fold_id=fold.fold_id,
                        seed=seed,
                        event_ids=train_tensors["event_id"].cpu().numpy().astype(np.int64),
                        t_us=train_tensors["t_us"].cpu().numpy().astype(np.int64),
                        supervised_indices=indices,
                        report=supervision_report(len(train_df), indices, cfg.supervision_mode, cfg.supervision_interval_us),
                    )

            test_supervised_mask: Optional[torch.Tensor] = None
            if cfg.eval_on_spine:
                test_indices = select_supervised_indices(
                    test_df,
                    mode=cfg.supervision_mode,
                    interval_us=cfg.supervision_interval_us,
                    include_large_events=cfg.include_large_events,
                    size_quantile=cfg.large_event_quantile,
                    every_n=cfg.supervision_every_n,
                )
                test_supervised_mask = build_supervision_mask(len(test_df), test_indices)

            model = build_model(train_cfg).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg.lr,
                weight_decay=train_cfg.weight_decay,
            )

            train_for_fit = train_tensors
            train_mask_for_fit = train_supervised_mask
            val_tensors: Optional[Dict[str, torch.Tensor]] = None
            val_mask: Optional[torch.Tensor] = None
            if cfg.early_stopping:
                train_inner_idx, val_idx = chronological_train_val_split_indices(
                    train_tensors,
                    val_fraction=cfg.val_fraction,
                    purge_us=cfg.embargo_us,
                    min_val_events=16,
                )
                if len(val_idx) > 0:
                    train_for_fit = prepare_sequence_tensors(subset_tensors(train_tensors, train_inner_idx))
                    val_tensors = prepare_sequence_tensors(subset_tensors(train_tensors, val_idx))
                    if train_supervised_mask is not None:
                        train_mask_for_fit = train_supervised_mask[torch.as_tensor(train_inner_idx, dtype=torch.long)]
                        val_mask = train_supervised_mask[torch.as_tensor(val_idx, dtype=torch.long)]

            # Train on this fold
            best_state: Optional[Dict[str, torch.Tensor]] = None
            best_score = float("inf")
            stale_epochs = 0
            for epoch in range(1, train_cfg.epochs + 1):
                train_one_epoch(
                    model=model,
                    tensors=train_for_fit,
                    optimizer=optimizer,
                    cfg=train_cfg,
                    device=device,
                    epoch=epoch,
                    supervised_mask=train_mask_for_fit,
                )
                if cfg.early_stopping and val_tensors is not None:
                    score, val_summary = validation_score(
                        model=model,
                        train_inner=train_for_fit,
                        val_inner=val_tensors,
                        cfg=cfg,
                        train_cfg=train_cfg,
                        device=device,
                        target_scaler=target_scaler,
                        supervised_mask=val_mask,
                    )
                    if score < best_score - cfg.min_delta:
                        best_score = score
                        best_state = copy.deepcopy(model.state_dict())
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                    print(
                        f"Validation seed={seed} fold={fold.fold_id} epoch={epoch:02d} | "
                        f"score={score:.6f} | spearman_10s={val_summary['log_rv_spearman_10s']:.4f}"
                    )
                    if stale_epochs >= cfg.patience:
                        print(f"Early stopping seed={seed} fold={fold.fold_id} at epoch {epoch:02d}")
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            # Rebuild consistent memory state under final trained weights
            replay_sequence_into_memory(
                model=model,
                tensors=train_tensors,
                device=device,
            )

            # Evaluate test fold sequentially
            fold_output = evaluate_sequence(
                model=model,
                tensors=test_tensors,
                cfg=cfg,
                device=device,
                supervised_mask=test_supervised_mask,
                target_scaler=target_scaler,
            )
            seed_fold_outputs.append(fold_output)

            fold_summary = summarize_seed_results([fold_output], cfg)
            fold_summary["seed"] = seed
            fold_summary["fold_id"] = fold.fold_id
            fold_summary["num_test_events"] = fold_output["num_events"]
            fold_rows.append(fold_summary)

            if prediction_dir is not None:
                pred_df = pd.DataFrame(
                    {
                        "seed": seed,
                        "fold_id": fold.fold_id,
                        "event_id": fold_output["event_id"],
                        "t_us": fold_output["t_us"],
                        "rv_1s_true": fold_output["vol_true"][:, 0],
                        "rv_5s_true": fold_output["vol_true"][:, 1],
                        "rv_10s_true": fold_output["vol_true"][:, 2],
                        "rv_1s_pred": fold_output["vol_pred"][:, 0],
                        "rv_5s_pred": fold_output["vol_pred"][:, 1],
                        "rv_10s_pred": fold_output["vol_pred"][:, 2],
                    }
                )
                if cfg.enable_price_move_head:
                    pred_df["price_move_true"] = fold_output["price_true"]
                    pred_df["price_move_pred"] = fold_output["price_pred"]
                pred_path = os.path.join(
                    prediction_dir,
                    f"ctgnn_seed_{seed}_fold_{fold.fold_id:03d}.parquet",
                )
                pred_df.to_parquet(pred_path, index=False)

        seed_summary = summarize_seed_results(seed_fold_outputs, cfg)
        seed_summary["seed"] = seed
        seed_rows.append(seed_summary)

    fold_df = pd.DataFrame(fold_rows)
    seed_df = pd.DataFrame(seed_rows)
    summary_df = summarize_across_seeds(seed_df)

    if cfg.out_dir is not None:
        fold_df.to_csv(os.path.join(cfg.out_dir, "fold_metrics.csv"), index=False)
        seed_df.to_csv(os.path.join(cfg.out_dir, "seed_metrics.csv"), index=False)
        summary_df.to_csv(os.path.join(cfg.out_dir, "summary_table.csv"), index=False)
        manifest = build_run_manifest(
            args=None,
            config=cfg,
            repo_root=Path(__file__).resolve().parents[1],
            device=cfg.device,
            metadata={
                "num_reconstructed_events": int(len(df)),
                "number_of_folds": int(len(folds)),
                "events_per_fold": [
                    {
                        "fold_id": int(f.fold_id),
                        "num_train": int(f.num_train),
                        "num_test": int(f.num_test),
                    }
                    for f in folds
                ],
                "seeds": list(cfg.seeds),
                "output_paths": {
                    "fold_metrics": os.path.join(cfg.out_dir, "fold_metrics.csv"),
                    "seed_metrics": os.path.join(cfg.out_dir, "seed_metrics.csv"),
                    "summary_table": os.path.join(cfg.out_dir, "summary_table.csv"),
                    "predictions": prediction_dir,
                    "scalers": scaler_dir,
                    "supervision": supervision_dir,
                },
                "elapsed_wall_clock_sec": time.time() - start_time,
                "peak_memory_gb": log_memory("after eval_runner"),
            },
        )
        save_run_manifest(Path(cfg.out_dir) / "run_manifest.json", manifest)

    return fold_df, seed_df, summary_df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_seeds(seeds_str: Optional[str], num_seeds: int, base_seed: int) -> Tuple[int, ...]:
    if seeds_str:
        vals = [int(x.strip()) for x in seeds_str.split(",") if x.strip()]
        if len(vals) < 1 or len(vals) > 5:
            raise ValueError("Provide between 1 and 5 seeds.")
        return tuple(vals)

    if num_seeds < 1 or num_seeds > 5:
        raise ValueError("num_seeds must be between 1 and 5.")
    return tuple(base_seed + i for i in range(num_seeds))


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Purged walk-forward evaluation suite for CTGNN.")
    parser.add_argument("--events", required=True, help="Path to events.parquet")
    parser.add_argument("--targets", required=True, help="Path to targets.parquet")
    parser.add_argument("--price-move-label-col", default=None)
    parser.add_argument("--feature-cols", default="", help="Comma-separated event feature columns.")

    parser.add_argument("--train-window-us", type=int, required=True)
    parser.add_argument("--test-window-us", type=int, required=True)
    parser.add_argument("--embargo-us", type=int, default=EMBARGO_US)
    parser.add_argument("--step-us", type=int, default=None)
    parser.add_argument("--anchored", action="store_true")
    parser.add_argument("--min-train-events", type=int, default=1_000)
    parser.add_argument("--min-test-events", type=int, default=1_000)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--mc-samples-train", type=int, default=None)
    parser.add_argument("--mc-samples-eval", type=int, default=None)
    parser.add_argument("--w-gap-nll", type=float, default=1.0)
    parser.add_argument("--w-event-type", type=float, default=1.0)
    parser.add_argument("--w-location", type=float, default=1.0)
    parser.add_argument("--w-volatility", type=float, default=1.0)
    parser.add_argument("--w-price-move", type=float, default=1.0)

    parser.add_argument("--enable-price-move-head", action="store_true")
    parser.add_argument("--no-log-rv-targets", action="store_true")
    parser.add_argument("--normalize-vol-targets", action="store_true")
    parser.add_argument("--target-scaler", choices=["standard"], default="standard")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-metric", choices=["val_loss", "val_spearman_10s", "composite"], default="composite")
    parser.add_argument("--lambda-rank", type=float, default=0.25)
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

    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=None)

    args = parser.parse_args()

    seeds = parse_seeds(args.seeds, args.num_seeds, args.base_seed)
    feature_cols = tuple(x.strip() for x in args.feature_cols.split(",") if x.strip()) or tuple(DEFAULT_FEATURE_COLS)

    return EvalConfig(
        events_path=args.events,
        targets_path=args.targets,
        price_move_label_col=args.price_move_label_col,
        feature_cols=feature_cols,
        train_window_us=args.train_window_us,
        test_window_us=args.test_window_us,
        embargo_us=args.embargo_us,
        step_us=args.step_us,
        anchored=args.anchored,
        min_train_events=args.min_train_events,
        min_test_events=args.min_test_events,
        seeds=seeds,
        use_log_rv_targets=not args.no_log_rv_targets,
        epochs=args.epochs,
        chunk_size=args.chunk_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        mc_samples=args.mc_samples,
        mc_samples_train=args.mc_samples_train,
        mc_samples_eval=args.mc_samples_eval,
        w_gap_nll=args.w_gap_nll,
        w_event_type=args.w_event_type,
        w_location=args.w_location,
        w_volatility=args.w_volatility,
        w_price_move=args.w_price_move,
        enable_price_move_head=args.enable_price_move_head,
        normalize_vol_targets=args.normalize_vol_targets,
        target_scaler=args.target_scaler,
        early_stopping=args.early_stopping,
        val_fraction=args.val_fraction,
        patience=args.patience,
        min_delta=args.min_delta,
        early_stopping_metric=args.early_stopping_metric,
        lambda_rank=args.lambda_rank,
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
        num_levels=args.num_levels,
        num_event_types=args.num_event_types,
        num_nodes=args.num_nodes,
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
        device=args.device,
        out_dir=args.out_dir,
    )


def main() -> None:
    cfg = parse_args()
    fold_df, seed_df, summary_df = run_purged_walk_forward_experiment(cfg)

    print("\n=== Seed-Level Metrics ===")
    print(seed_df.to_string(index=False))

    print("\n=== Final Summary ===")
    if summary_df.empty:
        print("No metrics produced (dry-run or no evaluated folds).")
    else:
        print(summary_df[["metric", "formatted"]].to_string(index=False))

    if cfg.out_dir is not None:
        print(f"\nSaved outputs to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
