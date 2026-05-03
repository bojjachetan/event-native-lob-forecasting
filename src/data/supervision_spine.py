from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch


SUPPORTED_MODES = {
    "all_events",
    "every_n_events",
    "every_100ms",
    "every_250ms",
    "every_500ms",
    "last_event_per_bucket",
    "volatility_informative",
}


def _interval_for_mode(mode: str, interval_us: Optional[int]) -> int:
    if mode == "every_100ms":
        return 100_000
    if mode == "every_250ms":
        return 250_000
    if mode == "every_500ms":
        return 500_000
    return int(interval_us or 250_000)


def _last_event_per_bucket(t_us: np.ndarray, interval_us: int, max_events_per_bucket: Optional[int]) -> np.ndarray:
    if len(t_us) == 0:
        return np.empty(0, dtype=np.int64)
    rel = t_us.astype(np.int64) - int(t_us[0])
    buckets = rel // int(interval_us)
    df = pd.DataFrame({"idx": np.arange(len(t_us), dtype=np.int64), "bucket": buckets})
    selected = df.groupby("bucket", sort=True)["idx"].tail(1).to_numpy(dtype=np.int64)
    if max_events_per_bucket is not None and max_events_per_bucket > 1:
        extra = []
        for _, group in df.groupby("bucket", sort=True):
            extra.extend(group["idx"].tail(max_events_per_bucket).to_numpy(dtype=np.int64).tolist())
        selected = np.unique(np.concatenate([selected, np.asarray(extra, dtype=np.int64)]))
    return np.sort(selected)


def select_supervised_indices(
    df: pd.DataFrame,
    mode: str = "all_events",
    interval_us: Optional[int] = None,
    max_events_per_bucket: Optional[int] = None,
    include_large_events: bool = False,
    size_quantile: float = 0.95,
    always_include_test_representatives: Optional[Iterable[int]] = None,
    every_n: int = 10,
) -> np.ndarray:
    """
    Select real event indices for expensive supervised losses/readouts.

    The spine never creates synthetic timestamps. All returned indices refer to
    original chronological events, so non-selected events can still be replayed
    causally through memory while gradients are spent only on selected events.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported supervision mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}")
    if len(df) == 0:
        return np.empty(0, dtype=np.int64)
    if "t_us" not in df.columns:
        raise KeyError("select_supervised_indices requires a t_us column.")

    n = len(df)
    if mode == "all_events":
        selected = np.arange(n, dtype=np.int64)
    elif mode == "every_n_events":
        selected = np.arange(0, n, max(int(every_n), 1), dtype=np.int64)
        selected = np.unique(np.concatenate([selected, np.asarray([n - 1], dtype=np.int64)]))
    else:
        interval = _interval_for_mode(mode, interval_us)
        selected = _last_event_per_bucket(
            df["t_us"].to_numpy(dtype=np.int64),
            interval_us=interval,
            max_events_per_bucket=max_events_per_bucket,
        )

    if include_large_events and "size" in df.columns and 0.0 <= size_quantile < 1.0:
        size = np.abs(df["size"].to_numpy(dtype=np.float64))
        if np.isfinite(size).any():
            threshold = float(np.nanquantile(size, size_quantile))
            large = np.flatnonzero(size >= threshold).astype(np.int64)
            selected = np.unique(np.concatenate([selected, large]))

    if mode == "volatility_informative":
        rv_cols = [c for c in ("rv_1s", "rv_5s", "rv_10s") if c in df.columns]
        if rv_cols:
            rv = df[rv_cols].to_numpy(dtype=np.float64)
            score = np.nanmean(np.abs(rv), axis=1)
            threshold = float(np.nanquantile(score, min(max(size_quantile, 0.0), 0.999)))
            selected = np.unique(np.concatenate([selected, np.flatnonzero(score >= threshold).astype(np.int64)]))

    if always_include_test_representatives is not None:
        reps = np.asarray(list(always_include_test_representatives), dtype=np.int64)
        reps = reps[(reps >= 0) & (reps < n)]
        selected = np.unique(np.concatenate([selected, reps]))

    return np.sort(selected.astype(np.int64))


def build_supervision_mask(num_events: int, indices: np.ndarray) -> torch.Tensor:
    mask = torch.zeros(int(num_events), dtype=torch.bool)
    if len(indices):
        mask[torch.as_tensor(indices, dtype=torch.long)] = True
    return mask


def supervision_report(num_events: int, supervised_indices: np.ndarray, mode: str, interval_us: Optional[int]) -> Dict[str, Any]:
    supervised = int(len(supervised_indices))
    return {
        "mode": mode,
        "interval_us": interval_us,
        "original_events": int(num_events),
        "supervised_events": supervised,
        "compression_ratio": float(num_events / max(supervised, 1)),
    }


def save_supervision_artifacts(
    out_dir: str | Path,
    *,
    fold_id: int,
    seed: int,
    event_ids: np.ndarray,
    t_us: np.ndarray,
    supervised_indices: np.ndarray,
    report: Dict[str, Any],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "seed": seed,
            "fold_id": fold_id,
            "local_index": supervised_indices.astype(np.int64),
            "event_id": event_ids[supervised_indices].astype(np.int64),
            "t_us": t_us[supervised_indices].astype(np.int64),
        }
    )
    df.to_parquet(out / f"supervised_indices_seed_{seed}_fold_{fold_id:03d}.parquet", index=False)
    with (out / f"supervision_spine_report_seed_{seed}_fold_{fold_id:03d}.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
