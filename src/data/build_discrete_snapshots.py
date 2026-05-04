
"""
build_discrete_snapshots.py

Converts continuous-event datasets:
  - events.parquet
  - targets.parquet
plus a purged walk-forward split manifest:
  - split_manifest.json / .csv / .parquet

into the discretized datasets needed by the baseline models:

1) DeepLOB adapter (100ms)
   - takes the last observed 40-dim top-10 LOB state in each 100ms bucket
   - builds rolling [N, T, 40] sequences within each fold/split
   - labels each sequence with the target of the representative continuous event

2) Static GCN adapter (1s)
   - takes the last observed state in each 1s bucket
   - builds [N, 20, F_node] node feature tensors
   - labels each snapshot with the target of the representative continuous event

Critical split alignment:
  - bucket labels are inherited from the exact last continuous event in the bucket
  - fold assignment uses the representative event timestamp
  - train/test samples are built separately within each purged fold, so they do not
    cross the embargo boundary or leak across fold edges

Outputs:
  out_dir/
    metadata/
      adapter_manifest.json
      split_manifest_copy.json
      static_gcn_edge_index.npy
    deeplob/
      fold_001_train.npz
      fold_001_test.npz
      ...
    static_gcn/
      fold_001_train.npz
      fold_001_test.npz
      ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


RV_COLS = ["rv_1s", "rv_5s", "rv_10s"]


# ---------------------------------------------------------------------
# Config / schema helpers
# ---------------------------------------------------------------------

@dataclass
class FoldRecord:
    fold_id: int
    train_start_us: int
    train_end_us: int
    embargo_start_us: int
    embargo_end_us: int
    test_start_us: int
    test_end_us: int


def bid_px_cols(num_levels: int) -> List[str]:
    return [f"bid_px_{i}" for i in range(1, num_levels + 1)]


def bid_sz_cols(num_levels: int) -> List[str]:
    return [f"bid_sz_{i}" for i in range(1, num_levels + 1)]


def ask_px_cols(num_levels: int) -> List[str]:
    return [f"ask_px_{i}" for i in range(1, num_levels + 1)]


def ask_sz_cols(num_levels: int) -> List[str]:
    return [f"ask_sz_{i}" for i in range(1, num_levels + 1)]


def lob_state_cols(num_levels: int) -> List[str]:
    cols: List[str] = []
    for i in range(1, num_levels + 1):
        cols.extend([f"bid_px_{i}", f"bid_sz_{i}", f"ask_px_{i}", f"ask_sz_{i}"])
    return cols


def required_event_cols(num_levels: int) -> List[str]:
    return ["event_id", "t_us", "mid", "spread"] + lob_state_cols(num_levels)


def required_target_cols() -> List[str]:
    return ["event_id"] + RV_COLS


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def load_split_manifest(path: str) -> List[FoldRecord]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        df = pd.DataFrame(rows)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("split_manifest must be .json, .csv, or .parquet")

    required = [
        "fold_id",
        "train_start_us",
        "train_end_us",
        "embargo_start_us",
        "embargo_end_us",
        "test_start_us",
        "test_end_us",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"split_manifest missing columns: {missing}")

    folds = []
    for _, row in df.sort_values("fold_id").iterrows():
        folds.append(
            FoldRecord(
                fold_id=int(row["fold_id"]),
                train_start_us=int(row["train_start_us"]),
                train_end_us=int(row["train_end_us"]),
                embargo_start_us=int(row["embargo_start_us"]),
                embargo_end_us=int(row["embargo_end_us"]),
                test_start_us=int(row["test_start_us"]),
                test_end_us=int(row["test_end_us"]),
            )
        )
    return folds


def load_and_merge(events_path: str, targets_path: str, num_levels: int) -> pd.DataFrame:
    events = pd.read_parquet(events_path)
    targets = pd.read_parquet(targets_path)

    missing_events = [c for c in required_event_cols(num_levels) if c not in events.columns]
    if missing_events:
        raise ValueError(f"events.parquet missing columns: {missing_events}")

    missing_targets = [c for c in required_target_cols() if c not in targets.columns]
    if missing_targets:
        raise ValueError(f"targets.parquet missing columns: {missing_targets}")

    df = events[required_event_cols(num_levels)].merge(
        targets[required_target_cols()],
        on="event_id",
        how="inner",
        validate="one_to_one",
    )

    df = df.reset_index(drop=False).rename(columns={"index": "_orig_order"})
    df = df.sort_values(["t_us", "_orig_order"], kind="mergesort").reset_index(drop=True)

    if df.empty:
        raise ValueError("Merged dataframe is empty.")
    if np.any(df["t_us"].to_numpy()[1:] < df["t_us"].to_numpy()[:-1]):
        raise ValueError("t_us must be sorted after merge.")

    return df


# ---------------------------------------------------------------------
# Continuous -> discrete bucket snapshots
# ---------------------------------------------------------------------

def make_bucket_snapshots(
    df: pd.DataFrame,
    bucket_us: int,
    num_levels: int,
    log_rv_eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Build one representative row per bucket by taking the *last observed event*
    in each bucket. The target is inherited from that exact representative event.

    This is the key alignment rule that keeps the baselines tied to exact
    continuous out-of-sample timestamps.
    """
    if bucket_us <= 0:
        raise ValueError("bucket_us must be positive.")

    out = df.copy()
    base_t = int(out["t_us"].iloc[0])
    out["bucket_id"] = ((out["t_us"].astype(np.int64) - base_t) // bucket_us).astype(np.int64)

    # Representative event = last event in the bucket
    snap = out.groupby("bucket_id", sort=True, as_index=False).tail(1).copy()
    snap = snap.sort_values(["t_us", "_orig_order"], kind="mergesort").reset_index(drop=True)

    snap["bucket_interval_us"] = int(bucket_us)
    snap["bucket_start_us"] = base_t + snap["bucket_id"].astype(np.int64) * int(bucket_us)
    snap["bucket_end_us"] = snap["bucket_start_us"] + int(bucket_us)

    # Add log-RV labels for convenience. Training/eval can choose which to use.
    for col in RV_COLS:
        snap[f"log_{col}"] = np.log(np.clip(snap[col].to_numpy(dtype=np.float64), a_min=log_rv_eps, a_max=None))

    ordered_cols = (
        ["event_id", "t_us", "bucket_id", "bucket_interval_us", "bucket_start_us", "bucket_end_us", "mid", "spread"]
        + lob_state_cols(num_levels)
        + RV_COLS
        + [f"log_{c}" for c in RV_COLS]
    )
    return snap[ordered_cols].copy()


# ---------------------------------------------------------------------
# Static GCN graph utilities
# ---------------------------------------------------------------------

def build_static_lob_edge_index_np(num_levels: int = 10, include_self_loops: bool = False) -> np.ndarray:
    """
    Nodes:
      0..num_levels-1            -> B1..BN
      num_levels..2*num_levels-1 -> A1..AN
    """
    src: List[int] = []
    dst: List[int] = []
    ask_offset = num_levels

    # Within-bid adjacency
    for i in range(num_levels - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])

    # Within-ask adjacency
    for i in range(num_levels - 1):
        u = ask_offset + i
        v = ask_offset + i + 1
        src.extend([u, v])
        dst.extend([v, u])

    # Cross-side same-distance coupling
    for i in range(num_levels):
        b = i
        a = ask_offset + i
        src.extend([b, a])
        dst.extend([a, b])

    if include_self_loops:
        for i in range(2 * num_levels):
            src.append(i)
            dst.append(i)

    return np.asarray([src, dst], dtype=np.int64)


def snapshot_row_to_gcn_node_features(
    row: pd.Series,
    num_levels: int,
) -> np.ndarray:
    """
    Converts one snapshot row into a [20, F_node] tensor.

    Node features:
      [price, size, rel_price_to_mid_bps, side_indicator, level_norm]

    where:
      side_indicator = +1 for bid, -1 for ask
      level_norm     = level_index / num_levels in (0,1]
    """
    mid = float(row["mid"])
    mid_safe = max(mid, 1e-12)

    feats: List[List[float]] = []

    for lvl in range(1, num_levels + 1):
        px = float(row[f"bid_px_{lvl}"])
        sz = float(row[f"bid_sz_{lvl}"])
        rel_bps = 1e4 * (px - mid) / mid_safe if px > 0 else 0.0
        feats.append([px, sz, rel_bps, 1.0, lvl / float(num_levels)])

    for lvl in range(1, num_levels + 1):
        px = float(row[f"ask_px_{lvl}"])
        sz = float(row[f"ask_sz_{lvl}"])
        rel_bps = 1e4 * (px - mid) / mid_safe if px > 0 else 0.0
        feats.append([px, sz, rel_bps, -1.0, lvl / float(num_levels)])

    return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------------
# Split alignment
# ---------------------------------------------------------------------

def split_snapshots_for_fold(
    snap: pd.DataFrame,
    fold: FoldRecord,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign discrete snapshots to the fold by the representative continuous-event timestamp.
    """
    t = snap["t_us"].to_numpy(dtype=np.int64)

    train_mask = (t >= fold.train_start_us) & (t < fold.train_end_us)
    test_mask = (t >= fold.test_start_us) & (t < fold.test_end_us)

    train_df = snap.loc[train_mask].copy().reset_index(drop=True)
    test_df = snap.loc[test_mask].copy().reset_index(drop=True)

    return train_df, test_df


# ---------------------------------------------------------------------
# DeepLOB adapter
# ---------------------------------------------------------------------

def snapshot_row_to_flat_lob40(row: pd.Series, num_levels: int) -> np.ndarray:
    """
    Returns the canonical 40-dim flattened LOB state:
      [bid_px_1..10, bid_sz_1..10, ask_px_1..10, ask_sz_1..10]
    """
    vec: List[float] = []
    for lvl in range(1, num_levels + 1):
        vec.append(float(row[f"bid_px_{lvl}"]))
    for lvl in range(1, num_levels + 1):
        vec.append(float(row[f"bid_sz_{lvl}"]))
    for lvl in range(1, num_levels + 1):
        vec.append(float(row[f"ask_px_{lvl}"]))
    for lvl in range(1, num_levels + 1):
        vec.append(float(row[f"ask_sz_{lvl}"]))
    return np.asarray(vec, dtype=np.float32)


def build_deeplob_sequences(
    snap_df: pd.DataFrame,
    num_levels: int,
    seq_len: int,
    use_log_targets: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Builds rolling [N, T, 40] samples inside one split only.
    No sequence is allowed to cross fold boundaries, because this function
    is called on already split-specific snapshot tables.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")

    n = len(snap_df)
    if n < seq_len:
        return {
            "X": np.empty((0, seq_len, 4 * num_levels), dtype=np.float32),
            "y": np.empty((0, 3), dtype=np.float32),
            "t_us": np.empty((0,), dtype=np.int64),
            "event_id": np.empty((0,), dtype=np.int64),
            "bucket_id": np.empty((0,), dtype=np.int64),
        }

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    t_list: List[int] = []
    eid_list: List[int] = []
    bucket_list: List[int] = []

    target_cols = [f"log_{c}" for c in RV_COLS] if use_log_targets else RV_COLS

    flat_states = np.stack(
        [snapshot_row_to_flat_lob40(row, num_levels) for _, row in snap_df.iterrows()],
        axis=0,
    )

    y_mat = snap_df[target_cols].to_numpy(dtype=np.float32)
    t_arr = snap_df["t_us"].to_numpy(dtype=np.int64)
    eid_arr = snap_df["event_id"].to_numpy(dtype=np.int64)
    bucket_arr = snap_df["bucket_id"].to_numpy(dtype=np.int64)

    for end_idx in range(seq_len - 1, n):
        start_idx = end_idx - seq_len + 1
        X_list.append(flat_states[start_idx : end_idx + 1])
        y_list.append(y_mat[end_idx])
        t_list.append(int(t_arr[end_idx]))
        eid_list.append(int(eid_arr[end_idx]))
        bucket_list.append(int(bucket_arr[end_idx]))

    return {
        "X": np.stack(X_list, axis=0).astype(np.float32),
        "y": np.stack(y_list, axis=0).astype(np.float32),
        "t_us": np.asarray(t_list, dtype=np.int64),
        "event_id": np.asarray(eid_list, dtype=np.int64),
        "bucket_id": np.asarray(bucket_list, dtype=np.int64),
    }


# ---------------------------------------------------------------------
# Static GCN adapter
# ---------------------------------------------------------------------

def build_static_gcn_snapshots(
    snap_df: pd.DataFrame,
    num_levels: int,
    use_log_targets: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Builds [N, 20, F_node] snapshot tensors inside one split only.
    """
    n = len(snap_df)
    if n == 0:
        return {
            "node_x": np.empty((0, 2 * num_levels, 5), dtype=np.float32),
            "y": np.empty((0, 3), dtype=np.float32),
            "t_us": np.empty((0,), dtype=np.int64),
            "event_id": np.empty((0,), dtype=np.int64),
            "bucket_id": np.empty((0,), dtype=np.int64),
        }

    target_cols = [f"log_{c}" for c in RV_COLS] if use_log_targets else RV_COLS

    node_x = np.stack(
        [snapshot_row_to_gcn_node_features(row, num_levels) for _, row in snap_df.iterrows()],
        axis=0,
    ).astype(np.float32)

    y = snap_df[target_cols].to_numpy(dtype=np.float32)
    t_us = snap_df["t_us"].to_numpy(dtype=np.int64)
    event_id = snap_df["event_id"].to_numpy(dtype=np.int64)
    bucket_id = snap_df["bucket_id"].to_numpy(dtype=np.int64)

    return {
        "node_x": node_x,
        "y": y,
        "t_us": t_us,
        "event_id": event_id,
        "bucket_id": bucket_id,
    }


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------

def save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def reset_output_dir(out_dir: str) -> None:
    for subdir in ("metadata", "deeplob", "static_gcn"):
        path = os.path.join(out_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path)


def serialize_fold_outputs(
    fold: FoldRecord,
    deeplob_train: Dict[str, np.ndarray],
    deeplob_test: Dict[str, np.ndarray],
    gcn_train: Dict[str, np.ndarray],
    gcn_test: Dict[str, np.ndarray],
    out_dir: str,
) -> None:
    fold_name = f"fold_{fold.fold_id:03d}"

    save_npz(os.path.join(out_dir, "deeplob", f"{fold_name}_train.npz"), deeplob_train)
    save_npz(os.path.join(out_dir, "deeplob", f"{fold_name}_test.npz"), deeplob_test)

    save_npz(os.path.join(out_dir, "static_gcn", f"{fold_name}_train.npz"), gcn_train)
    save_npz(os.path.join(out_dir, "static_gcn", f"{fold_name}_test.npz"), gcn_test)


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------

def build_discrete_baseline_datasets(
    events_path: str,
    targets_path: str,
    split_manifest_path: str,
    out_dir: str,
    num_levels: int = 10,
    deeplob_bucket_us: int = 100_000,     # 100ms
    gcn_bucket_us: int = 1_000_000,       # 1s
    deeplob_seq_len: int = 50,
    use_log_targets: bool = True,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    reset_output_dir(out_dir)
    os.makedirs(os.path.join(out_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "deeplob"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "static_gcn"), exist_ok=True)

    folds = load_split_manifest(split_manifest_path)
    merged = load_and_merge(events_path, targets_path, num_levels=num_levels)

    snap_100ms = make_bucket_snapshots(
        merged,
        bucket_us=deeplob_bucket_us,
        num_levels=num_levels,
    )
    snap_1s = make_bucket_snapshots(
        merged,
        bucket_us=gcn_bucket_us,
        num_levels=num_levels,
    )

    edge_index = build_static_lob_edge_index_np(num_levels=num_levels, include_self_loops=False)
    np.save(os.path.join(out_dir, "metadata", "static_gcn_edge_index.npy"), edge_index)

    # Copy split manifest for exact provenance
    if split_manifest_path.endswith(".json"):
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest_rows = json.load(f)
    else:
        manifest_rows = [asdict(fold) for fold in folds]
    with open(os.path.join(out_dir, "metadata", "split_manifest_copy.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)

    fold_stats: List[Dict[str, object]] = []

    for fold in folds:
        dtrain_df, dtest_df = split_snapshots_for_fold(snap_100ms, fold)
        gtrain_df, gtest_df = split_snapshots_for_fold(snap_1s, fold)

        deeplob_train = build_deeplob_sequences(
            dtrain_df,
            num_levels=num_levels,
            seq_len=deeplob_seq_len,
            use_log_targets=use_log_targets,
        )
        deeplob_test = build_deeplob_sequences(
            dtest_df,
            num_levels=num_levels,
            seq_len=deeplob_seq_len,
            use_log_targets=use_log_targets,
        )

        gcn_train = build_static_gcn_snapshots(
            gtrain_df,
            num_levels=num_levels,
            use_log_targets=use_log_targets,
        )
        gcn_test = build_static_gcn_snapshots(
            gtest_df,
            num_levels=num_levels,
            use_log_targets=use_log_targets,
        )

        serialize_fold_outputs(
            fold=fold,
            deeplob_train=deeplob_train,
            deeplob_test=deeplob_test,
            gcn_train=gcn_train,
            gcn_test=gcn_test,
            out_dir=out_dir,
        )

        fold_stats.append(
            {
                "fold_id": fold.fold_id,
                "deeplob_train_samples": int(deeplob_train["X"].shape[0]),
                "deeplob_test_samples": int(deeplob_test["X"].shape[0]),
                "gcn_train_samples": int(gcn_train["node_x"].shape[0]),
                "gcn_test_samples": int(gcn_test["node_x"].shape[0]),
                "deeplob_train_first_t_us": int(deeplob_train["t_us"][0]) if deeplob_train["t_us"].size else None,
                "deeplob_test_first_t_us": int(deeplob_test["t_us"][0]) if deeplob_test["t_us"].size else None,
                "gcn_train_first_t_us": int(gcn_train["t_us"][0]) if gcn_train["t_us"].size else None,
                "gcn_test_first_t_us": int(gcn_test["t_us"][0]) if gcn_test["t_us"].size else None,
            }
        )

    manifest = {
        "num_levels": num_levels,
        "deeplob_bucket_us": deeplob_bucket_us,
        "gcn_bucket_us": gcn_bucket_us,
        "deeplob_seq_len": deeplob_seq_len,
        "use_log_targets": use_log_targets,
        "notes": [
            "Each discrete sample inherits its target from the exact representative continuous event (last event in bucket).",
            "Fold assignment is based on the representative event timestamp t_us.",
            "DeepLOB sequences are built separately within each fold/split, so they never cross purge boundaries.",
            "Static GCN snapshots use 20 visible nodes only (10 bids + 10 asks); no execution sink node is used.",
        ],
        "fold_stats": fold_stats,
    }

    with open(os.path.join(out_dir, "metadata", "adapter_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build discrete snapshot datasets for DeepLOB and Static GCN baselines.")
    parser.add_argument("--events", required=True, help="Path to events.parquet")
    parser.add_argument("--targets", required=True, help="Path to targets.parquet")
    parser.add_argument("--split-manifest", required=True, help="Path to split_manifest.json/.csv/.parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory")

    parser.add_argument("--num-levels", type=int, default=10)
    parser.add_argument("--deeplob-bucket-us", type=int, default=100_000)
    parser.add_argument("--gcn-bucket-us", type=int, default=1_000_000)
    parser.add_argument("--deeplob-seq-len", type=int, default=50)

    parser.add_argument("--raw-rv-targets", action="store_true", help="Use raw RV targets instead of log-RV")
    args = parser.parse_args()

    manifest = build_discrete_baseline_datasets(
        events_path=args.events,
        targets_path=args.targets,
        split_manifest_path=args.split_manifest,
        out_dir=args.out_dir,
        num_levels=args.num_levels,
        deeplob_bucket_us=args.deeplob_bucket_us,
        gcn_bucket_us=args.gcn_bucket_us,
        deeplob_seq_len=args.deeplob_seq_len,
        use_log_targets=not args.raw_rv_targets,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
