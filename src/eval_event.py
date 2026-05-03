from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from src.utils.logging import format_mean_std, write_json
except ImportError:  # pragma: no cover
    from utils.logging import format_mean_std, write_json


HORIZONS = ("1s", "5s", "10s")


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


def prediction_files(path: str) -> List[Path]:
    base = Path(path)
    candidates = []
    if (base / "predictions").exists():
        candidates.extend(sorted((base / "predictions").glob("*.parquet")))
    candidates.extend(sorted(base.glob("*.parquet")))
    files = sorted({p.resolve() for p in candidates})
    if not files:
        raise FileNotFoundError(f"No prediction parquet files found under {path}")
    return [Path(p) for p in files]


def load_prediction_dir(path: str, model_name: str) -> pd.DataFrame:
    dfs = []
    for file_path in prediction_files(path):
        df = pd.read_parquet(file_path)
        if df.empty:
            continue
        df["model_name"] = model_name
        dfs.append(df)

    if not dfs:
        raise ValueError(f"Prediction directory {path} contained no non-empty parquet files.")

    out = pd.concat(dfs, axis=0, ignore_index=True)
    required = {"seed", "fold_id", "event_id", "t_us"}
    for horizon in HORIZONS:
        required.add(f"rv_{horizon}_true")
        required.add(f"rv_{horizon}_pred")
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Prediction directory {path} is missing columns: {missing}")

    out = out.sort_values(["seed", "fold_id", "t_us", "event_id"], kind="mergesort").reset_index(drop=True)
    return out


def metrics_from_frame(df: pd.DataFrame, pred_suffix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {"num_samples": float(len(df))}
    for horizon in HORIZONS:
        true_col = f"rv_{horizon}_true{pred_suffix}"
        pred_col = f"rv_{horizon}_pred{pred_suffix}"
        y_true = df[true_col].to_numpy(dtype=np.float64)
        y_pred = df[pred_col].to_numpy(dtype=np.float64)
        metrics[f"rmse_{horizon}"] = rmse(y_true, y_pred)
        metrics[f"mae_{horizon}"] = mae(y_true, y_pred)
        metrics[f"spearman_{horizon}"] = safe_spearman(y_pred, y_true)
    return metrics


def per_seed_metrics(
    df: pd.DataFrame,
    model_label: str,
    evaluation_set: str,
    pred_suffix: str = "",
) -> pd.DataFrame:
    rows = []
    for seed, group in df.groupby("seed", sort=True):
        row = {
            "seed": int(seed),
            "model": model_label,
            "evaluation_set": evaluation_set,
        }
        row.update(metrics_from_frame(group, pred_suffix=pred_suffix))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_seeds(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in per_seed_df.columns
        if c not in {"seed", "model", "evaluation_set"}
    ]
    rows = []
    for (evaluation_set, model), group in per_seed_df.groupby(["evaluation_set", "model"], sort=True):
        summary = {
            "evaluation_set": evaluation_set,
            "model": model,
        }
        for metric in metric_cols:
            vals = group[metric].to_numpy(dtype=np.float64)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary[f"{metric}_mean"] = mean_val
            summary[f"{metric}_std"] = std_val
            summary[f"{metric}_formatted"] = format_mean_std(mean_val, std_val)
        rows.append(summary)
    return pd.DataFrame(rows)


def align_predictions(
    ctgnn_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    baseline_name: str,
) -> pd.DataFrame:
    joined = baseline_df.merge(
        ctgnn_df,
        on=["seed", "fold_id", "event_id", "t_us"],
        suffixes=("_baseline", "_ctgnn"),
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError(f"No aligned events found for baseline {baseline_name}.")

    for horizon in HORIZONS:
        true_baseline = joined[f"rv_{horizon}_true_baseline"].to_numpy(dtype=np.float64)
        true_ctgnn = joined[f"rv_{horizon}_true_ctgnn"].to_numpy(dtype=np.float64)
        if not np.allclose(true_baseline, true_ctgnn, atol=1e-6, rtol=1e-8):
            raise ValueError(
                f"Aligned target mismatch for {baseline_name} at horizon {horizon}. "
                "This breaks timestamp integrity."
            )
        joined[f"rv_{horizon}_true"] = true_baseline
        joined[f"rv_{horizon}_pred_baseline"] = joined[f"rv_{horizon}_pred_baseline"].to_numpy(dtype=np.float64)
        joined[f"rv_{horizon}_pred_ctgnn"] = joined[f"rv_{horizon}_pred_ctgnn"].to_numpy(dtype=np.float64)

    return joined


def add_model_metrics(
    rows: List[pd.DataFrame],
    df: pd.DataFrame,
    evaluation_set: str,
    baseline_label: Optional[str] = None,
) -> None:
    if baseline_label is None:
        rows.append(per_seed_metrics(df, model_label="CTGNN", evaluation_set=evaluation_set))
        return

    ctgnn_frame = df[["seed"]].copy()
    baseline_frame = df[["seed"]].copy()
    for horizon in HORIZONS:
        ctgnn_frame[f"rv_{horizon}_true_ctgnn"] = df[f"rv_{horizon}_true"]
        ctgnn_frame[f"rv_{horizon}_pred_ctgnn"] = df[f"rv_{horizon}_pred_ctgnn"]
        baseline_frame[f"rv_{horizon}_true_baseline"] = df[f"rv_{horizon}_true"]
        baseline_frame[f"rv_{horizon}_pred_baseline"] = df[f"rv_{horizon}_pred_baseline"]

    rows.append(
        per_seed_metrics(
            ctgnn_frame,
            model_label="CTGNN",
            evaluation_set=evaluation_set,
            pred_suffix="_ctgnn",
        )
    )
    rows.append(
        per_seed_metrics(
            baseline_frame,
            model_label=baseline_label,
            evaluation_set=evaluation_set,
            pred_suffix="_baseline",
        )
    )


def build_neurips_tables(
    ctgnn_dir: str,
    deeplob_dir: Optional[str],
    static_gcn_dir: Optional[str],
    out_dir: str,
) -> Dict[str, object]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ctgnn_df = load_prediction_dir(ctgnn_dir, model_name="ctgnn")
    per_seed_frames: List[pd.DataFrame] = []

    per_seed_frames.append(
        per_seed_metrics(
            ctgnn_df,
            model_label="CTGNN",
            evaluation_set="all_event_times",
        )
    )

    if deeplob_dir is not None:
        deeplob_df = load_prediction_dir(deeplob_dir, model_name="deeplob")
        joined = align_predictions(ctgnn_df, deeplob_df, baseline_name="DeepLOB")
        joined.to_parquet(out_path / "aligned_deeplob_predictions.parquet", index=False)
        add_model_metrics(per_seed_frames, joined, evaluation_set="deeplob_aligned_times", baseline_label="DeepLOB")

    if static_gcn_dir is not None:
        static_gcn_df = load_prediction_dir(static_gcn_dir, model_name="static_gcn")
        joined = align_predictions(ctgnn_df, static_gcn_df, baseline_name="StaticGCN")
        joined.to_parquet(out_path / "aligned_static_gcn_predictions.parquet", index=False)
        add_model_metrics(per_seed_frames, joined, evaluation_set="static_gcn_aligned_times", baseline_label="StaticGCN")

    per_seed_df = pd.concat(per_seed_frames, axis=0, ignore_index=True)
    summary_df = summarize_across_seeds(per_seed_df)

    per_seed_df.to_csv(out_path / "per_seed_metrics.csv", index=False)
    summary_df.to_csv(out_path / "summary_table.csv", index=False)

    summary_payload = {
        "evaluation_sets": sorted(summary_df["evaluation_set"].unique().tolist()),
        "models": sorted(summary_df["model"].unique().tolist()),
        "summary_rows": summary_df.to_dict(orient="records"),
    }
    write_json(str(out_path / "summary_table.json"), summary_payload)
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned result tables from saved prediction files.")
    parser.add_argument("--ctgnn-dir", required=True, help="CTGNN evaluation output directory.")
    parser.add_argument("--deeplob-dir", default=None, help="DeepLOB output directory.")
    parser.add_argument("--static-gcn-dir", default=None, help="Static GCN output directory.")
    parser.add_argument("--out-dir", required=True, help="Directory for aligned summary tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_neurips_tables(
        ctgnn_dir=args.ctgnn_dir,
        deeplob_dir=args.deeplob_dir,
        static_gcn_dir=args.static_gcn_dir,
        out_dir=args.out_dir,
    )
    print(pd.DataFrame(summary["summary_rows"]).to_string(index=False))


if __name__ == "__main__":
    main()
