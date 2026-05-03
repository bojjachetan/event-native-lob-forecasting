from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.models.simple_baselines import PersistenceBaseline, RidgeBaseline, RollingMeanBaseline
from src.train import DEFAULT_FEATURE_COLS, RV_TARGET_COLS
from src.utils.run_manifest import build_run_manifest, save_run_manifest


HORIZONS = ("1s", "5s", "10s")
RIDGE_CLEAN = "RidgeClean"
RIDGE_SHUFFLED_Y = "RidgeShuffledY"
RIDGE_TIMESTAMP_PERMUTED = "RidgeTimestampPermuted"
FORBIDDEN_RIDGE_EXACT_COLUMNS = {"rv_1s", "rv_5s", "rv_10s"}
FORBIDDEN_RIDGE_SUBSTRINGS = (
    "log_rv",
    "future_return",
    "future_mid",
    "target_end",
    "horizon",
)
FORBIDDEN_RIDGE_PREFIXES = ("rv_",)


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".json":
        return pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported table format: {p}")


def _safe_spearman(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if len(y_pred) == 0 or np.allclose(y_pred, y_pred[0]) or np.allclose(y_true, y_true[0]):
        return 0.0
    val = spearmanr(y_pred, y_true).correlation
    return 0.0 if val is None or np.isnan(val) else float(val)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, h in enumerate(HORIZONS):
        diff = y_pred[:, i] - y_true[:, i]
        out[f"rmse_{h}"] = float(np.sqrt(np.mean(diff**2)))
        out[f"mae_{h}"] = float(np.mean(np.abs(diff)))
        out[f"spearman_{h}"] = _safe_spearman(y_pred[:, i], y_true[:, i])
    return out


def _ridge_feature_violation(col: str) -> str | None:
    name = col.lower()
    if name in FORBIDDEN_RIDGE_EXACT_COLUMNS:
        return "exact target column"
    if any(name.startswith(prefix) for prefix in FORBIDDEN_RIDGE_PREFIXES):
        return "rv/horizon-derived prefix"
    for token in FORBIDDEN_RIDGE_SUBSTRINGS:
        if token in name:
            return f"forbidden token '{token}'"
    return None


def assert_ridge_feature_columns_are_safe(cols: Sequence[str]) -> None:
    violations = [{"column": col, "reason": reason} for col in cols if (reason := _ridge_feature_violation(col))]
    if violations:
        raise AssertionError(f"Ridge feature leakage risk: {violations}")


def _feature_columns(events: pd.DataFrame, feature_cols: Tuple[str, ...], num_levels: int) -> List[str]:
    cols = [c for c in feature_cols if c in events.columns]
    for side in ("bid", "ask"):
        for kind in ("px", "sz"):
            cols.extend([f"{side}_{kind}_{i}" for i in range(1, num_levels + 1) if f"{side}_{kind}_{i}" in events.columns])
    if not cols:
        raise ValueError("No feature columns available for simple baselines.")
    # Preserve order while avoiding accidental duplicate columns from user-provided feature lists.
    cols = list(dict.fromkeys(cols))
    assert_ridge_feature_columns_are_safe(cols)
    return cols


def _feature_matrix(events: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    x = events[cols].to_numpy(dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _sha256_int64(values: Iterable[int]) -> str:
    arr = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float | str]]]:
    df = pd.DataFrame(rows)
    summary: Dict[str, Dict[str, Dict[str, float | str]]] = {}
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        model_summary: Dict[str, Dict[str, float | str]] = {}
        for col in [c for c in mdf.columns if c not in {"model", "fold_id"}]:
            vals = mdf[col].dropna().to_numpy(dtype=np.float64)
            mean = float(vals.mean()) if len(vals) else float("nan")
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            model_summary[col] = {"mean": mean, "std": std, "formatted": f"{mean:.4f} ± {std:.4f}"}
        summary[model] = model_summary
    return summary


def _ridge_fold_feature_audit(
    *,
    fold_id: int,
    feature_columns: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    ridge_model: RidgeBaseline,
    x: np.ndarray,
) -> Dict[str, object]:
    if ridge_model.x_mean_ is None or ridge_model.x_std_ is None:
        raise AssertionError("Ridge scaler statistics were not fitted.")
    train_x = x[train_idx].astype(np.float64)
    expected_mean = train_x.mean(axis=0)
    expected_std = train_x.std(axis=0)
    expected_std[expected_std < 1e-8] = 1.0
    np.testing.assert_allclose(ridge_model.x_mean_, expected_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ridge_model.x_std_, expected_std, rtol=1e-9, atol=1e-9)

    train_event_ids = df.iloc[train_idx]["event_id"].to_numpy(dtype=np.int64)
    test_event_ids = df.iloc[test_idx]["event_id"].to_numpy(dtype=np.int64)
    overlap = int(len(np.intersect1d(train_event_ids, test_event_ids)))
    if overlap:
        raise AssertionError(f"Ridge fold {fold_id} train/test event overlap: {overlap}")
    return {
        "fold_id": int(fold_id),
        "feature_columns": list(feature_columns),
        "num_features": int(len(feature_columns)),
        "num_train_rows": int(len(train_idx)),
        "num_test_rows": int(len(test_idx)),
        "scaler_fit_scope": "train_fold_only",
        "scaler_feature_count": int(len(ridge_model.x_mean_)),
        "train_event_id_sha256": _sha256_int64(train_event_ids),
        "test_event_id_sha256": _sha256_int64(test_event_ids),
        "train_t_us_min": int(df.iloc[train_idx]["t_us"].min()),
        "train_t_us_max": int(df.iloc[train_idx]["t_us"].max()),
        "test_t_us_min": int(df.iloc[test_idx]["t_us"].min()),
        "test_t_us_max": int(df.iloc[test_idx]["t_us"].max()),
    }


def _append_prediction(
    predictions: List[pd.DataFrame],
    *,
    model_name: str,
    fold_id: int,
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
) -> None:
    predictions.append(
        pd.DataFrame(
            {
                "model": model_name,
                "fold_id": fold_id,
                "event_id": df.iloc[test_idx]["event_id"].to_numpy(dtype=np.int64),
                "t_us": df.iloc[test_idx]["t_us"].to_numpy(dtype=np.int64),
                "rv_1s_true": y_true[:, 0],
                "rv_5s_true": y_true[:, 1],
                "rv_10s_true": y_true[:, 2],
                "rv_1s_pred": pred[:, 0],
                "rv_5s_pred": pred[:, 1],
                "rv_10s_pred": pred[:, 2],
            }
        )
    )


def run_simple_baselines(args: argparse.Namespace) -> Dict[str, object]:
    events = pd.read_parquet(args.events)
    targets = pd.read_parquet(args.targets)
    splits = _read_table(args.split_manifest)
    df = events.merge(targets[["event_id", *RV_TARGET_COLS]], on="event_id", how="inner", validate="one_to_one")
    df = df.sort_values("t_us", kind="mergesort").reset_index(drop=True)
    if args.use_log_targets:
        df[list(RV_TARGET_COLS)] = np.log(np.clip(df[list(RV_TARGET_COLS)].to_numpy(dtype=np.float64), args.log_eps, None))
    feature_columns = _feature_columns(df, tuple(args.feature_cols.split(",")), args.num_levels)
    x = _feature_matrix(df, feature_columns)
    y = df[list(RV_TARGET_COLS)].to_numpy(dtype=np.float32)

    fold_rows: List[Dict[str, float]] = []
    ridge_rows: List[Dict[str, float]] = []
    ridge_feature_audit_rows: List[Dict[str, object]] = []
    predictions = []
    for fold in splits.to_dict(orient="records"):
        fid = int(fold["fold_id"])
        train_idx = np.arange(int(fold["train_left"]), int(fold["train_right"]))
        test_idx = np.arange(int(fold["test_left"]), int(fold["test_right"]))
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        print(
            f"Ridge feature audit fold={fid}: {len(feature_columns)} columns: {', '.join(feature_columns)}",
            flush=True,
        )
        models = {
            "Persistence": PersistenceBaseline().fit(y[train_idx]),
            "RollingMean": RollingMeanBaseline(window=args.rolling_window).fit(y[train_idx]),
            RIDGE_CLEAN: RidgeBaseline(alpha=args.ridge_alpha).fit(x[train_idx], y[train_idx]),
        }
        for name, model in models.items():
            if name == RIDGE_CLEAN:
                pred = model.predict(x[test_idx])  # type: ignore[union-attr]
                ridge_feature_audit_rows.append(
                    _ridge_fold_feature_audit(
                        fold_id=fid,
                        feature_columns=feature_columns,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        df=df,
                        ridge_model=model,  # type: ignore[arg-type]
                        x=x,
                    )
                )
            else:
                pred = model.predict(len(test_idx))  # type: ignore[union-attr]
            m = _metrics(y[test_idx], pred)
            m["model"] = name  # type: ignore[assignment]
            m["fold_id"] = fid  # type: ignore[assignment]
            fold_rows.append(m)  # type: ignore[arg-type]
            if name == RIDGE_CLEAN:
                ridge_rows.append(m.copy())  # type: ignore[arg-type]
            _append_prediction(predictions, model_name=name, fold_id=fid, df=df, test_idx=test_idx, y_true=y[test_idx], pred=pred)

        if args.run_ridge_controls:
            rng = np.random.default_rng(int(args.control_seed) + fid * 9973)

            shuffled_y = y[train_idx][rng.permutation(len(train_idx))]
            shuffled_model = RidgeBaseline(alpha=args.ridge_alpha).fit(x[train_idx], shuffled_y)
            shuffled_pred = shuffled_model.predict(x[test_idx])
            shuffled_metrics = _metrics(y[test_idx], shuffled_pred)
            shuffled_metrics["model"] = RIDGE_SHUFFLED_Y  # type: ignore[assignment]
            shuffled_metrics["fold_id"] = fid  # type: ignore[assignment]
            fold_rows.append(shuffled_metrics)  # type: ignore[arg-type]
            ridge_rows.append(shuffled_metrics.copy())  # type: ignore[arg-type]
            _append_prediction(
                predictions,
                model_name=RIDGE_SHUFFLED_Y,
                fold_id=fid,
                df=df,
                test_idx=test_idx,
                y_true=y[test_idx],
                pred=shuffled_pred,
            )

            # This breaks the event-time pairing on both train and test sides while
            # preserving marginal feature/target distributions inside the same fold.
            train_time_perm = rng.permutation(len(train_idx))
            test_time_perm = rng.permutation(len(test_idx))
            timestamp_model = RidgeBaseline(alpha=args.ridge_alpha).fit(x[train_idx][train_time_perm], y[train_idx])
            timestamp_pred = timestamp_model.predict(x[test_idx][test_time_perm])
            timestamp_metrics = _metrics(y[test_idx], timestamp_pred)
            timestamp_metrics["model"] = RIDGE_TIMESTAMP_PERMUTED  # type: ignore[assignment]
            timestamp_metrics["fold_id"] = fid  # type: ignore[assignment]
            fold_rows.append(timestamp_metrics)  # type: ignore[arg-type]
            ridge_rows.append(timestamp_metrics.copy())  # type: ignore[arg-type]
            _append_prediction(
                predictions,
                model_name=RIDGE_TIMESTAMP_PERMUTED,
                fold_id=fid,
                df=df,
                test_idx=test_idx,
                y_true=y[test_idx],
                pred=timestamp_pred,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "simple_baselines_fold_metrics.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(out_dir / "simple_baselines_predictions.parquet", index=False)
    summary = {"model_name": "simple_baselines", "overall_summary": _summarize(fold_rows)}
    with (out_dir / "simple_baselines_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    ridge_feature_audit = {
        "status": "PASS",
        "feature_columns": feature_columns,
        "forbidden_exact_columns": sorted(FORBIDDEN_RIDGE_EXACT_COLUMNS),
        "forbidden_substrings": list(FORBIDDEN_RIDGE_SUBSTRINGS),
        "forbidden_prefixes": list(FORBIDDEN_RIDGE_PREFIXES),
        "folds": ridge_feature_audit_rows,
    }
    with (out_dir / "ridge_feature_audit.json").open("w", encoding="utf-8") as f:
        json.dump(ridge_feature_audit, f, indent=2)
    ridge_audit_summary = {
        "status": "PASS",
        "model_name": "ridge_audit",
        "feature_audit_path": str(out_dir / "ridge_feature_audit.json"),
        "negative_controls": {
            RIDGE_SHUFFLED_Y: "Train Ridge on train-fold targets randomly permuted across train-fold rows.",
            RIDGE_TIMESTAMP_PERMUTED: "Break event-time pairing by permuting train/test feature timestamps inside each fold.",
        },
        "overall_summary": _summarize(ridge_rows),
        "fold_metrics": ridge_rows,
    }
    with (out_dir / "ridge_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(ridge_audit_summary, f, indent=2)
    manifest = build_run_manifest(args=args, config=vars(args), repo_root=Path(__file__).resolve().parents[1], device="cpu")
    save_run_manifest(out_dir / "run_manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strict simple baselines on the same purged folds.")
    parser.add_argument("--events", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-cols", default=",".join(DEFAULT_FEATURE_COLS))
    parser.add_argument("--num-levels", type=int, default=10)
    parser.add_argument("--rolling-window", type=int, default=256)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--run-ridge-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-seed", type=int, default=20240201)
    parser.add_argument("--use-log-targets", action="store_true", default=True)
    parser.add_argument("--log-eps", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    summary = run_simple_baselines(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
