# make_splits.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd


EMBARGO_US = 300_000_000  # 5 minutes in microseconds


@dataclass
class PurgedWalkForwardFold:
    fold_id: int
    train_start_us: int
    train_end_us: int          # exclusive
    embargo_start_us: int
    embargo_end_us: int        # exclusive
    test_start_us: int
    test_end_us: int           # exclusive
    train_left: int
    train_right: int           # exclusive
    test_left: int
    test_right: int            # exclusive

    @property
    def train_indices(self) -> np.ndarray:
        return np.arange(self.train_left, self.train_right, dtype=np.int64)

    @property
    def test_indices(self) -> np.ndarray:
        return np.arange(self.test_left, self.test_right, dtype=np.int64)

    @property
    def num_train(self) -> int:
        return self.train_right - self.train_left

    @property
    def num_test(self) -> int:
        return self.test_right - self.test_left


def _validate_sorted_timestamps(t_us: np.ndarray) -> np.ndarray:
    t_us = np.asarray(t_us, dtype=np.int64)
    if t_us.ndim != 1:
        raise ValueError("t_us must be a 1D array.")
    if t_us.size == 0:
        raise ValueError("t_us is empty.")
    if np.any(t_us[1:] < t_us[:-1]):
        raise ValueError("t_us must be sorted in non-decreasing order.")
    return t_us


def generate_purged_walk_forward_splits(
    t_us: np.ndarray,
    train_window_us: int,
    test_window_us: int,
    embargo_us: int = EMBARGO_US,
    step_us: Optional[int] = None,
    anchored: bool = False,
    min_train_events: int = 1_000,
    min_test_events: int = 1_000,
) -> Iterator[PurgedWalkForwardFold]:
    """
    Strict purged walk-forward generator with half-open windows.

    Non-anchored:
      train = [train_start, train_start + train_window)
      purge = [train_end, train_end + embargo)
      test  = [test_start, test_start + test_window)

    Anchored:
      train start is fixed at t_us[0], train_end expands by step_us each fold.

    Notes:
      - Endpoints are exclusive.
      - The embargo guarantees zero temporal overlap between train and test.
      - step_us defaults to test_window_us.
    """
    t_us = _validate_sorted_timestamps(t_us)

    if train_window_us <= 0:
        raise ValueError("train_window_us must be positive.")
    if test_window_us <= 0:
        raise ValueError("test_window_us must be positive.")
    if embargo_us < 0:
        raise ValueError("embargo_us must be non-negative.")

    if step_us is None:
        step_us = test_window_us
    if step_us <= 0:
        raise ValueError("step_us must be positive.")

    global_start = int(t_us[0])
    global_end_exclusive = int(t_us[-1]) + 1

    fold_id = 1

    if anchored:
        train_start_us = global_start
        train_end_us = global_start + train_window_us

        while True:
            embargo_start_us = train_end_us
            embargo_end_us = embargo_start_us + embargo_us
            test_start_us = embargo_end_us
            test_end_us = test_start_us + test_window_us

            if test_start_us >= global_end_exclusive:
                break

            train_left = int(np.searchsorted(t_us, train_start_us, side="left"))
            train_right = int(np.searchsorted(t_us, train_end_us, side="left"))
            test_left = int(np.searchsorted(t_us, test_start_us, side="left"))
            test_right = int(np.searchsorted(t_us, test_end_us, side="left"))

            if (train_right - train_left) >= min_train_events and (test_right - test_left) >= min_test_events:
                yield PurgedWalkForwardFold(
                    fold_id=fold_id,
                    train_start_us=train_start_us,
                    train_end_us=train_end_us,
                    embargo_start_us=embargo_start_us,
                    embargo_end_us=embargo_end_us,
                    test_start_us=test_start_us,
                    test_end_us=test_end_us,
                    train_left=train_left,
                    train_right=train_right,
                    test_left=test_left,
                    test_right=test_right,
                )
                fold_id += 1

            train_end_us += step_us

    else:
        train_start_us = global_start

        while True:
            train_end_us = train_start_us + train_window_us
            embargo_start_us = train_end_us
            embargo_end_us = embargo_start_us + embargo_us
            test_start_us = embargo_end_us
            test_end_us = test_start_us + test_window_us

            if test_start_us >= global_end_exclusive:
                break

            train_left = int(np.searchsorted(t_us, train_start_us, side="left"))
            train_right = int(np.searchsorted(t_us, train_end_us, side="left"))
            test_left = int(np.searchsorted(t_us, test_start_us, side="left"))
            test_right = int(np.searchsorted(t_us, test_end_us, side="left"))

            if (train_right - train_left) >= min_train_events and (test_right - test_left) >= min_test_events:
                yield PurgedWalkForwardFold(
                    fold_id=fold_id,
                    train_start_us=train_start_us,
                    train_end_us=train_end_us,
                    embargo_start_us=embargo_start_us,
                    embargo_end_us=embargo_end_us,
                    test_start_us=test_start_us,
                    test_end_us=test_end_us,
                    train_left=train_left,
                    train_right=train_right,
                    test_left=test_left,
                    test_right=test_right,
                )
                fold_id += 1

            train_start_us += step_us


def build_split_manifest(
    t_us: np.ndarray,
    train_window_us: int,
    test_window_us: int,
    embargo_us: int = EMBARGO_US,
    step_us: Optional[int] = None,
    anchored: bool = False,
    min_train_events: int = 1_000,
    min_test_events: int = 1_000,
) -> pd.DataFrame:
    folds = list(
        generate_purged_walk_forward_splits(
            t_us=t_us,
            train_window_us=train_window_us,
            test_window_us=test_window_us,
            embargo_us=embargo_us,
            step_us=step_us,
            anchored=anchored,
            min_train_events=min_train_events,
            min_test_events=min_test_events,
        )
    )

    if not folds:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "train_start_us",
                "train_end_us",
                "embargo_start_us",
                "embargo_end_us",
                "test_start_us",
                "test_end_us",
                "train_left",
                "train_right",
                "test_left",
                "test_right",
                "num_train",
                "num_test",
            ]
        )

    rows = []
    for f in folds:
        row = asdict(f)
        row["num_train"] = f.num_train
        row["num_test"] = f.num_test
        rows.append(row)

    return pd.DataFrame(rows)


def _read_timestamps_from_file(path: str) -> np.ndarray:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=["t_us"])
    elif path.endswith(".csv"):
        df = pd.read_csv(path, usecols=["t_us"])
    else:
        raise ValueError("Supported input formats: .parquet or .csv")
    return df["t_us"].to_numpy(dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strict purged walk-forward split manifest.")
    parser.add_argument("--input", required=True, help="Parquet or CSV file containing a t_us column.")
    parser.add_argument("--out", required=True, help="Output manifest path (.parquet, .csv, or .json).")
    parser.add_argument("--train-window-us", type=int, required=True)
    parser.add_argument("--test-window-us", type=int, required=True)
    parser.add_argument("--embargo-us", type=int, default=EMBARGO_US)
    parser.add_argument("--step-us", type=int, default=None)
    parser.add_argument("--anchored", action="store_true")
    parser.add_argument("--min-train-events", type=int, default=1_000)
    parser.add_argument("--min-test-events", type=int, default=1_000)
    args = parser.parse_args()

    t_us = _read_timestamps_from_file(args.input)

    manifest = build_split_manifest(
        t_us=t_us,
        train_window_us=args.train_window_us,
        test_window_us=args.test_window_us,
        embargo_us=args.embargo_us,
        step_us=args.step_us,
        anchored=args.anchored,
        min_train_events=args.min_train_events,
        min_test_events=args.min_test_events,
    )

    if args.out.endswith(".parquet"):
        manifest.to_parquet(args.out, index=False)
    elif args.out.endswith(".csv"):
        manifest.to_csv(args.out, index=False)
    elif args.out.endswith(".json"):
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(orient="records"), f, indent=2)
    else:
        raise ValueError("Output must end with .parquet, .csv, or .json")

    print(f"Saved {len(manifest)} purged folds to {args.out}")


if __name__ == "__main__":
    main()