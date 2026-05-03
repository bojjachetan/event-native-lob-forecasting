# build_targets.py
from __future__ import annotations

import argparse
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_HORIZONS_S: tuple[int, ...] = (1, 5, 10)


def _required_state_columns(top_n: int) -> list[str]:
    cols = ["t_us", "event_type", "side", "level", "price", "size", "mid", "spread"]
    for i in range(1, top_n + 1):
        cols.append(f"bid_px_{i}")
        cols.append(f"bid_sz_{i}")
        cols.append(f"ask_px_{i}")
        cols.append(f"ask_sz_{i}")
    return cols


def validate_state_frame(df: pd.DataFrame, top_n: int = 10) -> None:
    missing = [c for c in _required_state_columns(top_n) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Input state DataFrame is empty.")

    if (df["mid"] <= 0).any():
        bad = int((df["mid"] <= 0).sum())
        raise ValueError(f"Found {bad} rows with non-positive mid-price.")

    if (df["spread"] < 0).any():
        bad = int((df["spread"] < 0).sum())
        raise ValueError(f"Found {bad} rows with negative spread.")


def prepare_state_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stable sort by exchange timestamp and preserve intra-timestamp order.
    """
    out = df.copy()
    # Allow repeated preparation during feature/target pipelines without
    # creating duplicate ordering columns.
    out = out.drop(columns=["_orig_order", "event_id"], errors="ignore")
    out = out.reset_index(drop=False).rename(columns={"index": "_orig_order"})
    out = out.sort_values(["t_us", "_orig_order"], kind="mergesort").reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=np.int64)
    return out


def _forward_realized_variance(
    t_us: np.ndarray,
    mid: np.ndarray,
    horizons_s: Sequence[int],
) -> dict[str, np.ndarray]:
    """
    Vectorized forward RV calculation from the true reconstructed mid-price path.

    For each event i and horizon H:
        RV_var(i, H) = sum_{k=i+1..j} (log_mid[k] - log_mid[k-1])^2
    where j is the largest index such that t_us[j] <= t_us[i] + H * 1e6.
    """
    if t_us.ndim != 1 or mid.ndim != 1:
        raise ValueError("t_us and mid must be 1D arrays.")
    if len(t_us) != len(mid):
        raise ValueError("t_us and mid must have the same length.")

    n = len(t_us)
    log_mid = np.log(mid.astype(np.float64))

    sq_log_ret = np.zeros(n, dtype=np.float64)
    if n > 1:
        dlog = np.diff(log_mid)
        sq_log_ret[1:] = dlog * dlog

    csum = np.cumsum(sq_log_ret)
    idx = np.arange(n, dtype=np.int64)

    out: dict[str, np.ndarray] = {}

    for h in horizons_s:
        horizon_us = int(h * 1_000_000)
        right = np.searchsorted(t_us, t_us + horizon_us, side="right") - 1
        right = np.maximum(right, idx)

        rv_var = csum[right] - csum[idx]
        rv_var = np.maximum(rv_var, 0.0)
        rv = np.sqrt(rv_var)

        out[f"rv_{h}s_var"] = rv_var
        out[f"rv_{h}s"] = rv
        out[f"rv_{h}s_end_t_us"] = t_us[right]
        out[f"rv_{h}s_num_returns"] = np.maximum(right - idx, 0)

    return out


def compute_forward_rv_targets(
    state_df: pd.DataFrame,
    top_n: int = 10,
    horizons_s: Sequence[int] = DEFAULT_HORIZONS_S,
) -> pd.DataFrame:
    validate_state_frame(state_df, top_n=top_n)
    df = prepare_state_frame(state_df)

    t_us = df["t_us"].to_numpy(dtype=np.int64)
    mid = df["mid"].to_numpy(dtype=np.float64)

    rv_dict = _forward_realized_variance(t_us=t_us, mid=mid, horizons_s=horizons_s)

    out = pd.DataFrame(
        {
            "event_id": df["event_id"].to_numpy(dtype=np.int64),
            "t_us": t_us,
            "mid": mid,
            "spread": df["spread"].to_numpy(dtype=np.float64),
        }
    )

    for k, v in rv_dict.items():
        out[k] = v

    return out


def serialize_targets_parquet(
    state_df: pd.DataFrame,
    out_path: str,
    top_n: int = 10,
    horizons_s: Sequence[int] = DEFAULT_HORIZONS_S,
) -> pd.DataFrame:
    targets = compute_forward_rv_targets(
        state_df=state_df,
        top_n=top_n,
        horizons_s=horizons_s,
    )
    targets.to_parquet(out_path, index=False)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute forward realized volatility targets from reconstructed LOB states.")
    parser.add_argument("--state", required=True, help="Input reconstructed state parquet")
    parser.add_argument("--out", required=True, help="Output targets parquet")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    state_df = pd.read_parquet(args.state)
    serialize_targets_parquet(
        state_df=state_df,
        out_path=args.out,
        top_n=args.top_n,
        horizons_s=DEFAULT_HORIZONS_S,
    )


if __name__ == "__main__":
    main()
