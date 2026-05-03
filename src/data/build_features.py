# build_features.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from src.data.build_targets import (
        DEFAULT_HORIZONS_S,
        compute_forward_rv_targets,
        prepare_state_frame,
        validate_state_frame,
    )
except ImportError:  # pragma: no cover
    from build_targets import (
        DEFAULT_HORIZONS_S,
        compute_forward_rv_targets,
        prepare_state_frame,
        validate_state_frame,
    )


EDGE_WITHIN_BID = 0
EDGE_WITHIN_ASK = 1
EDGE_CROSS_SAME_DISTANCE = 2

EVENT_TYPE_TO_CODE = {
    "add": 0,
    "cancel": 1,
    "execute": 2,
}

SIDE_TO_CODE = {
    "bid": 0,
    "ask": 1,
}


@dataclass
class GraphSpec:
    num_nodes: int
    edge_index: np.ndarray   # shape [2, E]
    edge_type: np.ndarray    # shape [E]
    node_names: list[str]
    edge_type_names: dict[int, str]


def build_static_lob_graph(top_n: int = 10) -> GraphSpec:
    """
    Static graph over visible top-N book:

      Nodes:
        0..top_n-1         => B1..BN
        top_n..2*top_n-1   => A1..AN

      Typed edges:
        0: within-bid adjacency
        1: within-ask adjacency
        2: cross-side same-distance coupling

    We add both directions for each conceptual adjacency.
    """
    src: list[int] = []
    dst: list[int] = []
    et: list[int] = []

    ask_offset = top_n

    # Within-bid adjacency: B1<->B2<->...<->BN
    for i in range(top_n - 1):
        u = i
        v = i + 1
        src.extend([u, v])
        dst.extend([v, u])
        et.extend([EDGE_WITHIN_BID, EDGE_WITHIN_BID])

    # Within-ask adjacency: A1<->A2<->...<->AN
    for i in range(top_n - 1):
        u = ask_offset + i
        v = ask_offset + i + 1
        src.extend([u, v])
        dst.extend([v, u])
        et.extend([EDGE_WITHIN_ASK, EDGE_WITHIN_ASK])

    # Cross-side same-distance coupling: Bk <-> Ak
    for i in range(top_n):
        b = i
        a = ask_offset + i
        src.extend([b, a])
        dst.extend([a, b])
        et.extend([EDGE_CROSS_SAME_DISTANCE, EDGE_CROSS_SAME_DISTANCE])

    edge_index = np.vstack(
        [
            np.asarray(src, dtype=np.int64),
            np.asarray(dst, dtype=np.int64),
        ]
    )
    edge_type = np.asarray(et, dtype=np.int8)

    node_names = [f"B{i}" for i in range(1, top_n + 1)] + [f"A{i}" for i in range(1, top_n + 1)]

    return GraphSpec(
        num_nodes=2 * top_n,
        edge_index=edge_index,
        edge_type=edge_type,
        node_names=node_names,
        edge_type_names={
            EDGE_WITHIN_BID: "within_bid_adjacency",
            EDGE_WITHIN_ASK: "within_ask_adjacency",
            EDGE_CROSS_SAME_DISTANCE: "cross_side_same_distance",
        },
    )


def _state_columns(top_n: int) -> tuple[list[str], list[str], list[str], list[str]]:
    bid_px_cols = [f"bid_px_{i}" for i in range(1, top_n + 1)]
    bid_sz_cols = [f"bid_sz_{i}" for i in range(1, top_n + 1)]
    ask_px_cols = [f"ask_px_{i}" for i in range(1, top_n + 1)]
    ask_sz_cols = [f"ask_sz_{i}" for i in range(1, top_n + 1)]
    return bid_px_cols, bid_sz_cols, ask_px_cols, ask_sz_cols


def build_event_feature_frame(
    state_df: pd.DataFrame,
    top_n: int = 10,
    keep_full_state: bool = True,
) -> pd.DataFrame:
    """
    Converts reconstructed LOB state rows into event rows suitable for a
    CTDG / TGN-style pipeline.

    Each row corresponds to one real exchange event affecting a visible top-N level.
    """
    validate_state_frame(state_df, top_n=top_n)
    df = prepare_state_frame(state_df)

    bid_px_cols, bid_sz_cols, ask_px_cols, ask_sz_cols = _state_columns(top_n)

    n = len(df)
    row_idx = np.arange(n, dtype=np.int64)

    event_type_code = df["event_type"].map(EVENT_TYPE_TO_CODE).to_numpy(dtype=np.int8)
    side_code = df["side"].map(SIDE_TO_CODE).to_numpy(dtype=np.int8)
    level = df["level"].to_numpy(dtype=np.int16)

    if np.any(level < 1) or np.any(level > top_n):
        raise ValueError(f"Found level outside [1, {top_n}] in visible-top-N dataset.")

    # Node mapping:
    #   bid level k => k-1
    #   ask level k => top_n + (k-1)
    node_id = np.where(side_code == 0, level - 1, top_n + (level - 1)).astype(np.int16)

    price = df["price"].to_numpy(dtype=np.float64)
    size = df["size"].to_numpy(dtype=np.float64)
    mid = df["mid"].to_numpy(dtype=np.float64)
    spread = df["spread"].to_numpy(dtype=np.float64)

    bid_px = df[bid_px_cols].to_numpy(dtype=np.float64)
    bid_sz = df[bid_sz_cols].to_numpy(dtype=np.float64)
    ask_px = df[ask_px_cols].to_numpy(dtype=np.float64)
    ask_sz = df[ask_sz_cols].to_numpy(dtype=np.float64)

    level0 = (level - 1).astype(np.int64)

    same_bid_px = bid_px[row_idx, level0]
    same_bid_sz = bid_sz[row_idx, level0]
    same_ask_px = ask_px[row_idx, level0]
    same_ask_sz = ask_sz[row_idx, level0]

    # Signed event size convention:
    #   add on bid   => +size
    #   add on ask   => -size
    #   remove on bid (cancel/execute) => -size
    #   remove on ask (cancel/execute) => +size
    is_add = (event_type_code == EVENT_TYPE_TO_CODE["add"]).astype(np.float64)
    event_dir = np.where(is_add > 0.5, 1.0, -1.0)
    side_dir = np.where(side_code == SIDE_TO_CODE["bid"], 1.0, -1.0)
    signed_event_size = side_dir * event_dir * size

    mid_safe = np.maximum(mid, 1e-12)
    rel_price_to_mid_bps = 1e4 * (price - mid) / mid_safe
    spread_bps = 1e4 * spread / mid_safe

    same_level_imbalance = (same_bid_sz - same_ask_sz) / np.maximum(same_bid_sz + same_ask_sz, 1e-12)
    visible_bid_depth = bid_sz.sum(axis=1)
    visible_ask_depth = ask_sz.sum(axis=1)
    visible_total_depth = np.maximum(visible_bid_depth + visible_ask_depth, 1e-12)

    current_visible_node_depth = np.where(side_code == SIDE_TO_CODE["bid"], same_bid_sz, same_ask_sz)
    node_depth_share = current_visible_node_depth / visible_total_depth

    best_bid = bid_px[:, 0]
    best_ask = ask_px[:, 0]
    book_imbalance_l1 = (bid_sz[:, 0] - ask_sz[:, 0]) / np.maximum(bid_sz[:, 0] + ask_sz[:, 0], 1e-12)

    event_df = pd.DataFrame(
        {
            "event_id": df["event_id"].to_numpy(dtype=np.int64),
            "t_us": df["t_us"].to_numpy(dtype=np.int64),
            "event_type": df["event_type"].astype("string"),
            "event_type_code": event_type_code,
            "side": df["side"].astype("string"),
            "side_code": side_code,
            "level": level,
            "node_id": node_id,
            "price": price,
            "size": size,
            "signed_event_size": signed_event_size,
            "mid": mid,
            "spread": spread,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "rel_price_to_mid_bps": rel_price_to_mid_bps,
            "spread_bps": spread_bps,
            "same_level_imbalance": same_level_imbalance,
            "book_imbalance_l1": book_imbalance_l1,
            "node_depth_share": node_depth_share,
            "visible_bid_depth": visible_bid_depth,
            "visible_ask_depth": visible_ask_depth,
        }
    )

    if keep_full_state:
        keep_cols = bid_px_cols + bid_sz_cols + ask_px_cols + ask_sz_cols
        for c in keep_cols:
            event_df[c] = df[c].to_numpy()

    return event_df


def serialize_lob_dataset(
    state_df: pd.DataFrame,
    events_out: str,
    targets_out: str,
    top_n: int = 10,
    horizons_s: Sequence[int] = DEFAULT_HORIZONS_S,
    keep_full_state: bool = True,
) -> GraphSpec:
    """
    Main entry point.

    Input:
      reconstructed LOB state rows with full visible book state

    Output:
      events.parquet
      targets.parquet

    Return:
      GraphSpec for the static top-N LOB graph
    """
    validate_state_frame(state_df, top_n=top_n)
    sorted_state = prepare_state_frame(state_df)

    events_df = build_event_feature_frame(
        state_df=sorted_state,
        top_n=top_n,
        keep_full_state=keep_full_state,
    )

    targets_df = compute_forward_rv_targets(
        state_df=sorted_state,
        top_n=top_n,
        horizons_s=horizons_s,
    )

    # Keep event_id as the join key between events.parquet and targets.parquet
    events_df.to_parquet(events_out, index=False)
    targets_df.to_parquet(targets_out, index=False)

    return build_static_lob_graph(top_n=top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CTDG event/features and forward RV targets from reconstructed LOB states.")
    parser.add_argument("--state", required=True, help="Input reconstructed state parquet")
    parser.add_argument("--events-out", required=True, help="Output events parquet")
    parser.add_argument("--targets-out", required=True, help="Output targets parquet")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    state_df = pd.read_parquet(args.state)

    graph = serialize_lob_dataset(
        state_df=state_df,
        events_out=args.events_out,
        targets_out=args.targets_out,
        top_n=args.top_n,
        horizons_s=DEFAULT_HORIZONS_S,
        keep_full_state=True,
    )

    print("Saved events and targets.")
    print(f"num_nodes={graph.num_nodes}")
    print(f"edge_index_shape={graph.edge_index.shape}")
    print(f"num_edges={graph.edge_index.shape[1]}")
    print(f"edge_type_names={graph.edge_type_names}")


if __name__ == "__main__":
    main()
