from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.memory import log_memory
from src.utils.run_manifest import build_run_manifest, save_run_manifest


FEATURE_COLS = (
    "signed_event_size",
    "size",
    "rel_price_to_mid_bps",
    "spread_bps",
    "same_level_imbalance",
    "book_imbalance_l1",
    "node_depth_share",
    "visible_bid_depth",
    "visible_ask_depth",
)


@dataclass
class TrialLayout:
    root_dir: Path
    artifact_root: Path
    download_dir: Path
    raw_dir: Path
    data_dir: Path
    output_dir: Path

    snapshot_path: Path
    depth_buffer_path: Path
    depth_path: Path
    trades_path: Path

    reconstructed_state_path: Path
    events_path: Path
    targets_path: Path
    split_manifest_path: Path

    ctgnn_dir: Path
    baseline_root: Path
    baseline_data_dir: Path
    deeplob_dir: Path
    static_gcn_dir: Path
    simple_baselines_dir: Path
    aligned_dir: Path
    audit_dir: Path
    metadata_path: Path


@dataclass
class DepthGroup:
    timestamp_us: int
    local_timestamp_us: int
    bids: List[List[str]]
    asks: List[List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Tardis BTCUSDT data, adapt it to the raw schema, and run the experiment pipeline.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", default="2020-02-01", help="Tardis trial date. First-of-month dates are public without an API key.")
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--trial-minutes", type=float, default=30.0, help="Keep only the first continuous segment of this duration after the initial snapshot.")
    parser.add_argument("--train-minutes", type=float, default=10.0)
    parser.add_argument("--test-minutes", type=float, default=5.0)
    parser.add_argument("--embargo-minutes", type=float, default=1.0)
    parser.add_argument("--step-minutes", type=float, default=5.0)
    parser.add_argument("--min-train-events", type=int, default=500)
    parser.add_argument("--min-test-events", type=int, default=200)
    parser.add_argument("--ctgnn-epochs", type=int, default=1)
    parser.add_argument("--baseline-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--mc-samples-train", type=int, default=None)
    parser.add_argument("--mc-samples-eval", type=int, default=None)
    parser.add_argument(
        "--train-on-spine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replay all events but compute supervised losses on a selected event spine.",
    )
    parser.add_argument("--eval-on-spine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--replay-all-events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--supervision-mode", default="all_events")
    parser.add_argument("--supervision-interval-us", type=int, default=250_000)
    parser.add_argument("--supervision-every-n", type=int, default=10)
    parser.add_argument("--include-large-events", action="store_true")
    parser.add_argument("--large-event-quantile", type=float, default=0.95)
    parser.add_argument("--normalize-vol-targets", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lambda-rank", type=float, default=0.25)
    parser.add_argument("--max-events-per-run", type=int, default=None)
    parser.add_argument("--dry-run-shapes", action="store_true")
    parser.add_argument("--progress-every-events", type=int, default=10_000)
    parser.add_argument("--deeplob-seq-len", type=int, default=20)
    parser.add_argument("--deeplob-bucket-us", type=int, default=100_000)
    parser.add_argument("--gcn-bucket-us", type=int, default=1_000_000)
    parser.add_argument("--baseline-purge-us", type=int, default=10_000_000)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-levels", type=int, default=5000)
    parser.add_argument("--match-window-us", type=int, default=250_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    return parser.parse_args()


def build_layout(root_dir: Path, symbol: str, date_str: str) -> TrialLayout:
    tag = f"tardis_trial_{symbol.lower()}_{date_str.replace('-', '')}"
    artifact_root = root_dir / "trial_runs" / tag
    download_dir = artifact_root / "downloads"
    raw_dir = artifact_root / "raw"
    data_dir = artifact_root / "data"
    output_dir = artifact_root / "outputs"

    return TrialLayout(
        root_dir=root_dir,
        artifact_root=artifact_root,
        download_dir=download_dir,
        raw_dir=raw_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        snapshot_path=raw_dir / f"{symbol}_snapshot.json.gz",
        depth_buffer_path=raw_dir / f"{symbol}_depth_buffer.jsonl.gz",
        depth_path=raw_dir / f"{symbol}_depth.jsonl.gz",
        trades_path=raw_dir / f"{symbol}_aggtrades.jsonl.gz",
        reconstructed_state_path=data_dir / "reconstructed_state.parquet",
        events_path=data_dir / "events.parquet",
        targets_path=data_dir / "targets.parquet",
        split_manifest_path=data_dir / "split_manifest.parquet",
        ctgnn_dir=output_dir / "eval" / "ctgnn",
        baseline_root=output_dir / "baselines",
        baseline_data_dir=output_dir / "baselines" / "data",
        deeplob_dir=output_dir / "baselines" / "deeplob",
        static_gcn_dir=output_dir / "baselines" / "static_gcn",
        simple_baselines_dir=output_dir / "baselines" / "simple",
        aligned_dir=output_dir / "aligned",
        audit_dir=output_dir / "audit",
        metadata_path=artifact_root / "trial_metadata.json",
    )


def ensure_dirs(layout: TrialLayout) -> None:
    for path in [
        layout.download_dir,
        layout.raw_dir,
        layout.data_dir,
        layout.ctgnn_dir,
        layout.baseline_data_dir,
        layout.deeplob_dir,
        layout.static_gcn_dir,
        layout.simple_baselines_dir,
        layout.aligned_dir,
        layout.audit_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def reset_pipeline_artifacts(layout: TrialLayout) -> None:
    """
    Clear derived trial artifacts so reruns do not mix fold files or metrics from
    previous executions. Raw downloaded inputs remain untouched.
    """
    for path in [layout.data_dir, layout.output_dir]:
        if path.exists():
            shutil.rmtree(path)
    ensure_dirs(layout)


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    return {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k is not None}


def pick(row: Dict[str, str], *keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    return default


def parse_bool(text: Optional[str]) -> bool:
    if text is None:
        return False
    return text.strip().lower() in {"1", "true", "t", "yes", "y"}


def normalize_book_side(text: str) -> Optional[str]:
    side = text.strip().lower()
    if side in {"bid", "buy"}:
        return "bid"
    if side in {"ask", "sell"}:
        return "ask"
    return None


def normalize_trade_side(text: str) -> Optional[str]:
    side = text.strip().lower()
    if side in {"buy", "sell"}:
        return side
    return None


def download_file(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "continuous-time-lob-trial/1.0"})
    with urllib.request.urlopen(req) as response, dest.open("wb") as f:
        shutil.copyfileobj(response, f)
    return dest


def tardis_csv_urls(exchange: str, symbol: str, date_str: str) -> Tuple[str, str]:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    prefix = f"https://datasets.tardis.dev/v1/{exchange}"
    day_path = f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{symbol}.csv.gz"
    return (
        f"{prefix}/incremental_book_L2/{day_path}",
        f"{prefix}/trades/{day_path}",
    )


def _sorted_book_side(levels: Dict[str, str], descending: bool) -> List[List[str]]:
    ordered = sorted(levels.items(), key=lambda kv: float(kv[0]), reverse=descending)
    return [[price, qty] for price, qty in ordered if float(qty) > 0.0]


def convert_incremental_l2_to_raw(
    input_path: Path,
    symbol: str,
    snapshot_out: Path,
    depth_buffer_out: Path,
    depth_out: Path,
    trial_minutes: int,
) -> Dict[str, object]:
    snapshot_levels: Dict[str, Dict[str, str]] = {"bid": {}, "ask": {}}
    update_groups: List[DepthGroup] = []
    first_update_ts: Optional[int] = None
    last_update_ts: Optional[int] = None
    stop_reason = "end_of_file"
    snapshot_rows = 0
    rows_read = 0
    update_rows = 0
    current_group_key: Optional[int] = None
    current_group_ts: Optional[int] = None
    current_bids: List[List[str]] = []
    current_asks: List[List[str]] = []
    seen_initial_snapshot = False
    in_updates = False
    cutoff_us = int(float(trial_minutes) * 60 * 1_000_000)

    def flush_group() -> None:
        if current_group_key is None or current_group_ts is None:
            return
        update_groups.append(
            DepthGroup(
                timestamp_us=current_group_ts,
                local_timestamp_us=current_group_key,
                bids=list(current_bids),
                asks=list(current_asks),
            )
        )

    with gzip.open(input_path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            rows_read += 1
            row = normalize_row(raw_row)
            timestamp_us = int(pick(row, "timestamp", default="0") or "0")
            local_timestamp_us = int(pick(row, "local_timestamp", default=str(timestamp_us)) or str(timestamp_us))
            side = normalize_book_side(str(pick(row, "side", default="")))
            price = pick(row, "price")
            amount = pick(row, "amount", "size", "qty")
            is_snapshot = parse_bool(pick(row, "is_snapshot", default="false"))

            if side is None or price is None or amount is None:
                continue

            if not seen_initial_snapshot:
                if not is_snapshot:
                    continue
                seen_initial_snapshot = True

            if seen_initial_snapshot and not in_updates:
                if is_snapshot:
                    snapshot_rows += 1
                    snapshot_levels[side][price] = amount
                    continue
                in_updates = True
                first_update_ts = timestamp_us

            if in_updates and is_snapshot:
                stop_reason = "encountered_snapshot_reset"
                break

            if first_update_ts is not None and timestamp_us - first_update_ts > cutoff_us:
                stop_reason = "reached_trial_duration_cutoff"
                break

            if current_group_key is None:
                current_group_key = local_timestamp_us
                current_group_ts = timestamp_us
            elif local_timestamp_us != current_group_key:
                flush_group()
                current_group_key = local_timestamp_us
                current_group_ts = timestamp_us
                current_bids = []
                current_asks = []

            update_rows += 1
            if side == "bid":
                current_bids.append([price, amount])
            else:
                current_asks.append([price, amount])
            last_update_ts = timestamp_us

    flush_group()

    if snapshot_rows == 0:
        raise RuntimeError("No initial Tardis snapshot rows were found in incremental_book_L2.")
    if not update_groups:
        raise RuntimeError("No post-snapshot incremental updates were found for the requested trial window.")

    snapshot_payload = {
        "lastUpdateId": 0,
        "bids": _sorted_book_side(snapshot_levels["bid"], descending=True),
        "asks": _sorted_book_side(snapshot_levels["ask"], descending=False),
    }
    with gzip.open(snapshot_out, "wt", encoding="utf-8") as f:
        json.dump(snapshot_payload, f)

    depth_events = []
    prev_u = 0
    for idx, group in enumerate(update_groups):
        if idx == 0:
            first_u = 0
            final_u = 1
        else:
            first_u = idx + 1
            final_u = idx + 1
        depth_events.append(
            {
                "e": "depthUpdate",
                "E": group.timestamp_us,
                "T": group.timestamp_us,
                "s": symbol,
                "U": first_u,
                "u": final_u,
                "pu": prev_u,
                "b": group.bids,
                "a": group.asks,
            }
        )
        prev_u = final_u

    with gzip.open(depth_buffer_out, "wt", encoding="utf-8") as f:
        f.write(json.dumps(depth_events[0]) + "\n")
    with gzip.open(depth_out, "wt", encoding="utf-8") as f:
        for event in depth_events[1:]:
            f.write(json.dumps(event) + "\n")

    return {
        "rows_read": rows_read,
        "snapshot_rows": snapshot_rows,
        "update_rows": update_rows,
        "depth_groups_written": len(depth_events),
        "trial_start_t_us": first_update_ts,
        "trial_end_t_us": last_update_ts,
        "stop_reason": stop_reason,
    }


def convert_trades_to_aggtrades(
    input_path: Path,
    symbol: str,
    output_path: Path,
    start_t_us: int,
    end_t_us: int,
) -> Dict[str, object]:
    rows_read = 0
    rows_written = 0
    skipped_unknown_side = 0
    next_trade_id = 1

    with gzip.open(input_path, "rt", newline="") as fin, gzip.open(output_path, "wt", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for raw_row in reader:
            rows_read += 1
            row = normalize_row(raw_row)
            timestamp_us = int(pick(row, "timestamp", default="0") or "0")
            if timestamp_us < start_t_us or timestamp_us > end_t_us:
                continue

            side = normalize_trade_side(str(pick(row, "side", "taker_side", default="")))
            price = pick(row, "price")
            amount = pick(row, "amount", "size", "qty")
            if price is None or amount is None:
                continue
            if side is None:
                skipped_unknown_side += 1
                continue

            trade_id_text = pick(row, "id", "trade_id")
            trade_id = int(trade_id_text) if trade_id_text not in {None, ""} else next_trade_id
            event = {
                "e": "aggTrade",
                "E": timestamp_us,
                "T": timestamp_us,
                "s": symbol,
                "a": trade_id,
                "p": price,
                "q": amount,
                "f": trade_id,
                "l": trade_id,
                "m": side == "sell",
            }
            fout.write(json.dumps(event) + "\n")
            rows_written += 1
            next_trade_id = max(next_trade_id, trade_id + 1)

    return {
        "rows_read": rows_read,
        "rows_written": rows_written,
        "skipped_unknown_side": skipped_unknown_side,
    }


def run_cmd(cmd: Sequence[str], root_dir: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root_dir}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    subprocess.run(list(cmd), cwd=root_dir, env=env, check=True)


def sanitize_trial_state(path: Path) -> Dict[str, int]:
    df = pd.read_parquet(path)
    before = int(len(df))
    negative_spread = int((df["spread"] < 0).sum())
    nonpositive_mid = int((df["mid"] <= 0).sum())
    keep_mask = (df["spread"] >= 0) & (df["mid"] > 0)
    cleaned = df.loc[keep_mask].reset_index(drop=True)
    cleaned.to_parquet(path, index=False)
    return {
        "rows_before": before,
        "rows_after": int(len(cleaned)),
        "negative_spread_rows_dropped": negative_spread,
        "nonpositive_mid_rows_dropped": nonpositive_mid,
    }


def write_metadata(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    start_time = time.time()
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    layout = build_layout(repo_root, args.symbol, args.date)
    ensure_dirs(layout)

    depth_url, trades_url = tardis_csv_urls(args.exchange, args.symbol, args.date)
    depth_csv_path = layout.download_dir / f"{args.symbol}_incremental_book_L2_{args.date}.csv.gz"
    trades_csv_path = layout.download_dir / f"{args.symbol}_trades_{args.date}.csv.gz"

    metadata: Dict[str, object] = {
        "trial_request": {
            "exchange": args.exchange,
            "symbol": args.symbol,
            "date": args.date,
            "trial_minutes": args.trial_minutes,
            "train_minutes": args.train_minutes,
            "test_minutes": args.test_minutes,
            "embargo_minutes": args.embargo_minutes,
            "step_minutes": args.step_minutes,
            "ctgnn_epochs": args.ctgnn_epochs,
            "baseline_epochs": args.baseline_epochs,
            "device": args.device,
            "seeds": args.seeds,
            "train_on_spine": args.train_on_spine,
            "supervision_mode": args.supervision_mode,
            "include_large_events": args.include_large_events,
            "mc_samples_train": args.mc_samples_train,
            "mc_samples_eval": args.mc_samples_eval,
            "normalize_vol_targets": args.normalize_vol_targets,
            "early_stopping": args.early_stopping,
        },
        "paths": {k: str(v) for k, v in asdict(layout).items()},
        "download_urls": {
            "incremental_book_L2": depth_url,
            "trades": trades_url,
        },
    }

    if not args.skip_download:
        download_file(depth_url, depth_csv_path)
        download_file(trades_url, trades_csv_path)

        depth_meta = convert_incremental_l2_to_raw(
            input_path=depth_csv_path,
            symbol=args.symbol,
            snapshot_out=layout.snapshot_path,
            depth_buffer_out=layout.depth_buffer_path,
            depth_out=layout.depth_path,
            trial_minutes=args.trial_minutes,
        )
        trade_meta = convert_trades_to_aggtrades(
            input_path=trades_csv_path,
            symbol=args.symbol,
            output_path=layout.trades_path,
            start_t_us=int(depth_meta["trial_start_t_us"]),
            end_t_us=int(depth_meta["trial_end_t_us"]),
        )
        metadata["adapted_depth"] = depth_meta
        metadata["adapted_trades"] = trade_meta
        write_metadata(layout.metadata_path, metadata)
    else:
        if not all(path.exists() for path in [layout.snapshot_path, layout.depth_buffer_path, layout.depth_path, layout.trades_path]):
            raise FileNotFoundError("skip-download was set, but the adapted raw trial files are missing.")

    if args.skip_pipeline:
        print(f"Prepared adapted raw files at: {layout.raw_dir}")
        print(f"Metadata written to: {layout.metadata_path}")
        return

    reset_pipeline_artifacts(layout)

    train_window_us = int(args.train_minutes * 60 * 1_000_000)
    test_window_us = int(args.test_minutes * 60 * 1_000_000)
    embargo_us = int(args.embargo_minutes * 60 * 1_000_000)
    step_us = int(args.step_minutes * 60 * 1_000_000)

    common_seed_args = ["--seeds", args.seeds]

    run_cmd(
        [
            args.python_bin,
            "-m",
            "src.data.reconstruct_lob",
            "--snapshot",
            str(layout.snapshot_path),
            "--depth-buffer",
            str(layout.depth_buffer_path),
            "--depth",
            str(layout.depth_path),
            "--trades",
            str(layout.trades_path),
            "--top-n",
            str(args.top_n),
            "--max-levels",
            str(args.max_levels),
            "--match-window-us",
            str(args.match_window_us),
            "--out",
            str(layout.reconstructed_state_path),
        ],
        repo_root,
    )
    metadata["trial_state_sanitation"] = sanitize_trial_state(layout.reconstructed_state_path)
    write_metadata(layout.metadata_path, metadata)

    run_cmd(
        [
            args.python_bin,
            "-m",
            "src.data.build_features",
            "--state",
            str(layout.reconstructed_state_path),
            "--events-out",
            str(layout.events_path),
            "--targets-out",
            str(layout.targets_path),
            "--top-n",
            str(args.top_n),
        ],
        repo_root,
    )

    run_cmd(
        [
            args.python_bin,
            "-m",
            "src.make_splits",
            "--input",
            str(layout.events_path),
            "--out",
            str(layout.split_manifest_path),
            "--train-window-us",
            str(train_window_us),
            "--test-window-us",
            str(test_window_us),
            "--embargo-us",
            str(embargo_us),
            "--step-us",
            str(step_us),
            "--min-train-events",
            str(args.min_train_events),
            "--min-test-events",
            str(args.min_test_events),
        ],
        repo_root,
    )

    eval_cmd = [
        args.python_bin,
        "-m",
        "src.eval_runner",
        "--events",
        str(layout.events_path),
        "--targets",
        str(layout.targets_path),
        "--feature-cols",
        ",".join(FEATURE_COLS),
        "--train-window-us",
        str(train_window_us),
        "--test-window-us",
        str(test_window_us),
        "--embargo-us",
        str(embargo_us),
        "--step-us",
        str(step_us),
        "--min-train-events",
        str(args.min_train_events),
        "--min-test-events",
        str(args.min_test_events),
        "--epochs",
        str(args.ctgnn_epochs),
        "--chunk-size",
        str(args.chunk_size),
        "--mc-samples",
        str(args.mc_samples),
        "--device",
        args.device,
        "--out-dir",
        str(layout.ctgnn_dir),
        *common_seed_args,
    ]
    if args.mc_samples_train is not None:
        eval_cmd += ["--mc-samples-train", str(args.mc_samples_train)]
    if args.mc_samples_eval is not None:
        eval_cmd += ["--mc-samples-eval", str(args.mc_samples_eval)]
    if args.train_on_spine:
        eval_cmd.append("--train-on-spine")
    if args.eval_on_spine:
        eval_cmd.append("--eval-on-spine")
    if not args.replay_all_events:
        eval_cmd.append("--no-replay-all-events")
    if args.include_large_events:
        eval_cmd.append("--include-large-events")
    if args.normalize_vol_targets:
        eval_cmd.append("--normalize-vol-targets")
    if args.early_stopping:
        eval_cmd.append("--early-stopping")
    if args.dry_run_shapes:
        eval_cmd.append("--dry-run-shapes")
    eval_cmd += [
        "--supervision-mode",
        args.supervision_mode,
        "--supervision-interval-us",
        str(args.supervision_interval_us),
        "--supervision-every-n",
        str(args.supervision_every_n),
        "--large-event-quantile",
        str(args.large_event_quantile),
        "--val-fraction",
        str(args.val_fraction),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--lambda-rank",
        str(args.lambda_rank),
        "--progress-every-events",
        str(args.progress_every_events),
    ]
    if args.max_events_per_run is not None:
        eval_cmd += ["--max-events-per-run", str(args.max_events_per_run)]
    run_cmd(eval_cmd, repo_root)

    run_cmd(
        [
            args.python_bin,
            "-m",
            "src.data.build_discrete_snapshots",
            "--events",
            str(layout.events_path),
            "--targets",
            str(layout.targets_path),
            "--split-manifest",
            str(layout.split_manifest_path),
            "--out-dir",
            str(layout.baseline_data_dir),
            "--num-levels",
            str(args.top_n),
            "--deeplob-bucket-us",
            str(args.deeplob_bucket_us),
            "--gcn-bucket-us",
            str(args.gcn_bucket_us),
            "--deeplob-seq-len",
            str(args.deeplob_seq_len),
        ],
        repo_root,
    )

    baselines_module = repo_root / "src" / "models" / "baselines.py"
    for model_name, out_dir in [("deeplob", layout.deeplob_dir), ("static_gcn", layout.static_gcn_dir)]:
        run_cmd(
            [
                args.python_bin,
                "-m",
                "src.train_baselines",
                "--data-dir",
                str(layout.baseline_data_dir),
                "--model-name",
                model_name,
                "--baselines-module-path",
                str(baselines_module),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.baseline_epochs),
                "--patience",
                "1",
                "--min-val-samples",
                "16",
                "--purge-us",
                str(args.baseline_purge_us),
                "--out-dir",
                str(out_dir),
                *common_seed_args,
            ],
            repo_root,
        )

    run_cmd(
        [
            args.python_bin,
            "-m",
            "src.train_simple_baselines",
            "--events",
            str(layout.events_path),
            "--targets",
            str(layout.targets_path),
            "--split-manifest",
            str(layout.split_manifest_path),
            "--out-dir",
            str(layout.simple_baselines_dir),
            "--feature-cols",
            ",".join(FEATURE_COLS),
            "--num-levels",
            str(args.top_n),
        ],
        repo_root,
    )

    run_cmd(
        [
            args.python_bin,
            "-m",
            "src.eval_event",
            "--ctgnn-dir",
            str(layout.ctgnn_dir),
            "--deeplob-dir",
            str(layout.deeplob_dir),
            "--static-gcn-dir",
            str(layout.static_gcn_dir),
            "--out-dir",
            str(layout.aligned_dir),
        ],
        repo_root,
    )

    summary_csv = layout.aligned_dir / "summary_table.csv"
    audit_cmd = [
        args.python_bin,
        "-m",
        "src.analysis.audit_experiment",
        "--events",
        str(layout.events_path),
        "--targets",
        str(layout.targets_path),
        "--split-manifest",
        str(layout.split_manifest_path),
        "--out-dir",
        str(layout.audit_dir),
        "--baseline-data-dir",
        str(layout.baseline_data_dir),
        "--aligned-summary",
        str(summary_csv),
        "--scaler-dir",
        str(layout.ctgnn_dir / "scalers"),
        "--seeds",
        args.seeds,
        "--ridge-audit-summary",
        str(layout.simple_baselines_dir / "ridge_audit_summary.json"),
    ]
    if args.normalize_vol_targets:
        # Audit distinguishes enabled-vs-disabled normalization; pass this only
        # when per-fold target scaler artifacts should exist.
        audit_cmd.append("--normalize-vol-targets")
        audit_cmd_note = "normalize_vol_targets enabled"
    else:
        audit_cmd_note = "normalize_vol_targets disabled"
    run_cmd(audit_cmd, repo_root)
    metadata["audit_target_normalization"] = audit_cmd_note
    metadata["pipeline"] = {
        "summary_csv": str(summary_csv),
        "ctgnn_dir": str(layout.ctgnn_dir),
        "deeplob_dir": str(layout.deeplob_dir),
        "static_gcn_dir": str(layout.static_gcn_dir),
        "simple_baselines_dir": str(layout.simple_baselines_dir),
        "aligned_dir": str(layout.aligned_dir),
        "audit_dir": str(layout.audit_dir),
    }
    write_metadata(layout.metadata_path, metadata)

    manifest = build_run_manifest(
        args=args,
        config=metadata,
        repo_root=repo_root,
        device=args.device,
        metadata={
            "dataset_date": args.date,
            "symbol": args.symbol,
            "trial_minutes": args.trial_minutes,
            "train_minutes": args.train_minutes,
            "test_minutes": args.test_minutes,
            "embargo_minutes": args.embargo_minutes,
            "step_minutes": args.step_minutes,
            "number_of_reconstructed_events": int(pd.read_parquet(layout.events_path, columns=["event_id"]).shape[0]),
            "number_of_folds": int(pd.read_parquet(layout.split_manifest_path).shape[0]),
            "seeds": [int(x.strip()) for x in args.seeds.split(",") if x.strip()],
            "output_paths": metadata["pipeline"],
            "elapsed_wall_clock_sec": time.time() - start_time,
            "peak_memory_gb": log_memory("after full trial pipeline"),
        },
    )
    save_run_manifest(layout.output_dir / "run_manifest.json", manifest)
    save_run_manifest(layout.artifact_root / "run_manifest.json", manifest)

    print(f"Trial pipeline completed. Summary table: {summary_csv}")


if __name__ == "__main__":
    main()
