from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


RV_COLS = ("rv_1s", "rv_5s", "rv_10s")
CORE_NUMERIC = (
    "t_us",
    "mid",
    "spread",
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


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise AssertionError(f"Required file does not exist: {p}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise AssertionError(f"Unsupported table format for {p}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _warning(warnings: List[str], message: str) -> None:
    warnings.append(message)


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def audit_events_targets(events: pd.DataFrame, targets: pd.DataFrame, warnings: List[str], num_levels: int) -> Dict[str, Any]:
    _assert(len(events) > 0, "events.parquet is empty")
    _assert(len(targets) > 0, "targets.parquet is empty")
    _assert("event_id" in events.columns and "event_id" in targets.columns, "event_id must exist in events and targets")
    _assert(events["event_id"].is_unique, "events.event_id is not unique")
    _assert(targets["event_id"].is_unique, "targets.event_id is not unique")
    joined = events[["event_id"]].merge(targets[["event_id"]], on="event_id", how="inner")
    _assert(len(joined) == len(events) == len(targets), "event_id join between events and targets is not one-to-one and complete")
    _assert("t_us" in events.columns, "events missing t_us")
    t_us = events["t_us"].to_numpy(dtype=np.int64)
    _assert(bool(np.all(t_us[1:] >= t_us[:-1])), "events.t_us is not sorted non-decreasing")

    numeric_cols = [c for c in CORE_NUMERIC if c in events.columns]
    numeric_cols += [c for c in RV_COLS if c in targets.columns]
    if numeric_cols:
        frame = pd.concat(
            [
                events[[c for c in numeric_cols if c in events.columns]],
                targets[[c for c in numeric_cols if c in targets.columns]],
            ],
            axis=1,
        )
        vals = frame.to_numpy(dtype=np.float64)
        _assert(bool(np.isfinite(vals).all()), "NaN/inf found in core numeric columns")

    for col in ("mid",):
        _assert(col in events.columns, f"events missing {col}")
    _assert(bool((events["mid"] > 0).all()), "mid must be positive")
    _assert("spread" in events.columns, "events missing spread")
    _assert(bool((events["spread"] >= 0).all()), "spread must be non-negative")

    bid_sz_cols = [f"bid_sz_{i}" for i in range(1, num_levels + 1)]
    ask_sz_cols = [f"ask_sz_{i}" for i in range(1, num_levels + 1)]
    bid_px_cols = [f"bid_px_{i}" for i in range(1, num_levels + 1)]
    ask_px_cols = [f"ask_px_{i}" for i in range(1, num_levels + 1)]
    required_lob_cols = bid_px_cols + bid_sz_cols + ask_px_cols + ask_sz_cols
    missing_lob = [c for c in required_lob_cols if c not in events.columns]
    _assert(not missing_lob, f"full post-event LOB state columns missing: {missing_lob}")
    _assert(bool((events[bid_sz_cols + ask_sz_cols] >= 0).all().all()), "bid/ask sizes must be non-negative")

    _assert("level" in events.columns, "events missing level")
    _assert(bool(events["level"].between(1, num_levels).all()), f"level must be in [1,{num_levels}]")
    _assert("side_code" in events.columns, "events missing side_code")
    _assert(set(events["side_code"].dropna().astype(int).unique()).issubset({0, 1}), "side_code must be in {0,1}")
    _assert("event_type_code" in events.columns, "events missing event_type_code")
    _assert(set(events["event_type_code"].dropna().astype(int).unique()).issubset({0, 1, 2}), "event_type_code must be in {0,1,2}")
    _assert("node_id" in events.columns, "events missing node_id")
    _assert(bool(events["node_id"].between(0, 2 * num_levels - 1).all()), "node_id outside visible book node range")

    return {
        "num_events": int(len(events)),
        "num_targets": int(len(targets)),
        "time_start_us": int(t_us[0]),
        "time_end_us": int(t_us[-1]),
    }


def audit_splits(events: pd.DataFrame, splits: pd.DataFrame, warnings: List[str]) -> Dict[str, Any]:
    required = [
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
    ]
    missing = [c for c in required if c not in splits.columns]
    _assert(not missing, f"split manifest missing columns: {missing}")
    _assert(len(splits) > 0, "split manifest has no folds")

    t_us = events["t_us"].to_numpy(dtype=np.int64)
    event_ids = events["event_id"].to_numpy(dtype=np.int64)
    fold_reports = []
    for row in splits.to_dict(orient="records"):
        fid = int(row["fold_id"])
        ts = int(row["train_start_us"])
        te = int(row["train_end_us"])
        es = int(row["embargo_start_us"])
        ee = int(row["embargo_end_us"])
        xs = int(row["test_start_us"])
        xe = int(row["test_end_us"])
        _assert(ts < te <= es < ee <= xs < xe, f"fold {fid} invalid window ordering")
        _assert(te <= es and ee <= xs, f"fold {fid} purge/embargo ordering invalid")
        _assert(te + (ee - es) <= xs, f"fold {fid} train_end + embargo_us exceeds test_start")

        train_left, train_right = int(row["train_left"]), int(row["train_right"])
        test_left, test_right = int(row["test_left"]), int(row["test_right"])
        train_t = t_us[train_left:train_right]
        test_t = t_us[test_left:test_right]
        train_ids = event_ids[train_left:train_right]
        test_ids = event_ids[test_left:test_right]
        _assert(bool(np.all((train_t >= ts) & (train_t < te))), f"fold {fid} train timestamps outside train window")
        _assert(bool(np.all((test_t >= xs) & (test_t < xe))), f"fold {fid} test timestamps outside test window")
        _assert(len(np.intersect1d(train_t, test_t)) == 0, f"fold {fid} train timestamp appears in test")
        _assert(len(np.intersect1d(train_ids, test_ids)) == 0, f"fold {fid} event assigned to both train and test")

        embargo_mask = (t_us >= es) & (t_us < ee)
        embargo_idx = np.flatnonzero(embargo_mask)
        train_idx = np.arange(train_left, train_right)
        test_idx = np.arange(test_left, test_right)
        _assert(len(np.intersect1d(embargo_idx, train_idx)) == 0, f"fold {fid} embargo event used in training")
        _assert(len(np.intersect1d(embargo_idx, test_idx)) == 0, f"fold {fid} embargo event used in testing")
        fold_reports.append({"fold_id": fid, "num_train": int(len(train_t)), "num_test": int(len(test_t))})
    return {"num_folds": int(len(splits)), "folds": fold_reports}


def _parse_seeds(seeds: Optional[str]) -> List[int]:
    if not seeds:
        return []
    return [int(x.strip()) for x in seeds.split(",") if x.strip()]


def audit_targets(
    targets: pd.DataFrame,
    events: pd.DataFrame,
    scaler_dir: Optional[Path],
    warnings: List[str],
    *,
    normalize_vol_targets: bool,
    expected_scaler_count: Optional[int] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "rv_columns": list(RV_COLS),
        "target_normalization": {
            "enabled": bool(normalize_vol_targets),
            "scaler_type": "standard" if normalize_vol_targets else None,
        },
    }
    for col in RV_COLS:
        _assert(col in targets.columns, f"targets missing {col}")
        _assert(bool(np.isfinite(targets[col].to_numpy(dtype=np.float64)).all()), f"{col} contains NaN/inf")
        end_col_candidates = [f"{col}_end_t_us", f"{col}_target_end_t_us", f"target_end_t_us_{col[-2:]}"]
        for end_col in end_col_candidates:
            if end_col in targets.columns:
                joined = targets[["event_id", end_col]].merge(events[["event_id", "t_us"]], on="event_id", how="inner")
                _assert(bool((joined[end_col] >= joined["t_us"]).all()), f"{end_col} is not forward-looking")
    if not normalize_vol_targets:
        report["target_normalization"].update(
            {
                "status": "disabled",
                "message": "normalize_vol_targets is disabled for this run; target scaler artifacts are not expected.",
            }
        )
        return report

    _assert(scaler_dir is not None and scaler_dir.exists(), "normalize_vol_targets is enabled but target scaler directory is missing")
    scaler_files = sorted(scaler_dir.glob("target_scaler*.json"))
    _assert(bool(scaler_files), f"normalize_vol_targets is enabled but no target_scaler*.json files were found in {scaler_dir}")
    if expected_scaler_count is not None:
        _assert(
            len(scaler_files) == expected_scaler_count,
            f"expected {expected_scaler_count} target scaler files, found {len(scaler_files)} in {scaler_dir}",
        )

    scaler_reports = []
    for path in scaler_files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        _assert(payload.get("fit_scope") == "train_fold_only", f"{path} does not declare fit_scope=train_fold_only")
        _assert(payload.get("type") == "standard", f"{path} target scaler type is not standard")
        mean = np.asarray(payload.get("mean", []), dtype=np.float64)
        std = np.asarray(payload.get("std", []), dtype=np.float64)
        _assert(mean.shape == (len(RV_COLS),), f"{path} mean must have shape [{len(RV_COLS)}]")
        _assert(std.shape == (len(RV_COLS),), f"{path} std must have shape [{len(RV_COLS)}]")
        _assert(bool(np.isfinite(mean).all() and np.isfinite(std).all()), f"{path} scaler mean/std contains NaN/inf")
        _assert(bool((std > 0).all()), f"{path} scaler std must be positive")
        scaler_reports.append(
            {
                "path": str(path),
                "seed": payload.get("seed"),
                "fold_id": payload.get("fold_id"),
                "fit_scope": payload.get("fit_scope"),
                "fit_event_count": payload.get("fit_event_count"),
            }
        )

    report["target_normalization"].update(
        {
            "status": "verified",
            "scaler_dir": str(scaler_dir),
            "num_scaler_files": int(len(scaler_files)),
            "expected_scaler_count": expected_scaler_count,
            "scalers": scaler_reports,
        }
    )
    return report


def _check_baseline_npz(path: Path, fold: Dict[str, Any], events: pd.DataFrame, model_name: str) -> Dict[str, int]:
    data = _load_npz(path)
    if data is None:
        return {"samples": 0}
    for key in ("event_id", "t_us", "y"):
        _assert(key in data, f"{path} missing {key}")
    ids = data["event_id"].astype(np.int64)
    t_us = data["t_us"].astype(np.int64)
    _assert(len(ids) == len(t_us) == data["y"].shape[0], f"{path} inconsistent sample dimensions")
    _assert(set(ids.tolist()).issubset(set(events["event_id"].astype(np.int64).tolist())), f"{path} representative event_id not in events")
    _assert(bool(np.all((t_us >= int(fold["test_start_us"])) & (t_us < int(fold["test_end_us"])))), f"{model_name} test sample outside CT-GNN test window in {path}")
    if model_name == "static_gcn":
        _assert("node_x" in data, f"{path} missing node_x")
        _assert(data["node_x"].shape[1] == 20, f"StaticGCN must use 20 visible nodes only, got {data['node_x'].shape[1]}")
    return {"samples": int(len(ids))}


def audit_baselines(
    events: pd.DataFrame,
    splits: pd.DataFrame,
    baseline_data_dir: Optional[Path],
    aligned_summary: Optional[Path],
    warnings: List[str],
) -> Dict[str, Any]:
    if baseline_data_dir is None or not baseline_data_dir.exists():
        _warning(warnings, "Baseline adapter outputs not found; skipped baseline alignment checks.")
        return {}

    out: Dict[str, Any] = {}
    manifest_path = baseline_data_dir / "metadata" / "adapter_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        notes = " ".join(manifest.get("notes", []))
        _assert("representative continuous event" in notes, "adapter manifest does not document representative continuous event alignment")
        _assert("never cross purge boundaries" in notes, "adapter manifest does not document DeepLOB sequence boundary isolation")
        _assert("20 visible nodes" in notes, "adapter manifest does not document StaticGCN visible-node-only design")
        out["adapter_manifest"] = manifest
    else:
        _warning(warnings, "No adapter_manifest.json found for baseline outputs.")

    counts: Dict[str, int] = {"deeplob_test": 0, "static_gcn_test": 0}
    for fold in splits.to_dict(orient="records"):
        fid = int(fold["fold_id"])
        for model in ("deeplob", "static_gcn"):
            path = baseline_data_dir / model / f"fold_{fid:03d}_test.npz"
            stats = _check_baseline_npz(path, fold, events, model)
            counts[f"{model}_test"] += stats.get("samples", 0)
    out["sample_counts"] = counts

    if aligned_summary is not None and aligned_summary.exists():
        summary = pd.read_csv(aligned_summary)
        for eval_set, count_key in (("deeplob_aligned_times", "deeplob_test"), ("static_gcn_aligned_times", "static_gcn_test")):
            if eval_set in set(summary.get("evaluation_set", [])):
                row = summary[summary["evaluation_set"] == eval_set].iloc[0]
                if "num_samples_mean" in row:
                    _assert(int(round(float(row["num_samples_mean"]))) == counts[count_key], f"aligned sample count mismatch for {eval_set}")
    return out


def audit_ridge_summary(ridge_audit_summary: Optional[Path], warnings: List[str]) -> Dict[str, Any]:
    if ridge_audit_summary is None or not ridge_audit_summary.exists():
        _warning(warnings, "Ridge audit summary not found; Ridge rows should not be treated as paper-clean.")
        return {"status": "missing"}
    with ridge_audit_summary.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    _assert(payload.get("status") == "PASS", f"Ridge audit did not pass: {ridge_audit_summary}")
    required_models = {"RidgeClean", "RidgeShuffledY", "RidgeTimestampPermuted"}
    present = set(payload.get("overall_summary", {}).keys())
    missing = sorted(required_models - present)
    _assert(not missing, f"Ridge audit summary missing required rows: {missing}")
    feature_audit_path = payload.get("feature_audit_path")
    if feature_audit_path:
        feature_path = Path(feature_audit_path)
        _assert(feature_path.exists(), f"Ridge feature audit file missing: {feature_path}")
        with feature_path.open("r", encoding="utf-8") as f:
            feature_payload = json.load(f)
        _assert(feature_payload.get("status") == "PASS", f"Ridge feature audit did not pass: {feature_path}")
    else:
        _warning(warnings, "Ridge audit summary does not reference ridge_feature_audit.json.")
    return {
        "status": "verified",
        "summary_path": str(ridge_audit_summary),
        "feature_audit_path": feature_audit_path,
        "models": sorted(present),
    }


def audit_masks(events: pd.DataFrame, warnings: List[str], num_levels: int) -> Dict[str, Any]:
    bid_sz = events[[f"bid_sz_{i}" for i in range(1, num_levels + 1)]].to_numpy(dtype=np.float32)
    ask_sz = events[[f"ask_sz_{i}" for i in range(1, num_levels + 1)]].to_numpy(dtype=np.float32)
    visible_populated = np.concatenate([bid_sz > 0, ask_sz > 0], axis=1)
    sink_present = True
    saved_mask_cols = [c for c in events.columns if c.startswith("node_populated_mask")]
    if saved_mask_cols:
        saved = events[saved_mask_cols].to_numpy(dtype=bool)
        expected = np.concatenate([visible_populated, np.ones((len(events), 1), dtype=bool)], axis=1)
        _assert(saved.shape == expected.shape and bool((saved == expected).all()), "saved node_populated_mask does not match size-derived mask")
    else:
        _warning(warnings, "No saved node_populated_mask columns found; recomputed mask from bid/ask sizes only.")
    return {"visible_nodes": 2 * num_levels, "sink_node_present": sink_present, "saved_mask_columns": saved_mask_cols}


def write_reports(report: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "audit_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    lines = [
        "# Experiment Audit Report",
        "",
        f"Status: **{report['status']}**",
        f"Events: {report.get('events_targets', {}).get('num_events', 'n/a')}",
        f"Folds: {report.get('splits', {}).get('num_folds', 'n/a')}",
        "",
        "## Warnings",
    ]
    if report["warnings"]:
        lines += [f"- {w}" for w in report["warnings"]]
    else:
        lines.append("- None")
    (out_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    warnings: List[str] = []
    events = _read_table(args.events)
    targets = _read_table(args.targets)
    splits = _read_table(args.split_manifest)
    scaler_dir = Path(args.scaler_dir) if args.scaler_dir else None
    baseline_data_dir = Path(args.baseline_data_dir) if args.baseline_data_dir else None
    aligned_summary = Path(args.aligned_summary) if args.aligned_summary else None
    ridge_audit_summary = Path(args.ridge_audit_summary) if args.ridge_audit_summary else None
    seeds = _parse_seeds(args.seeds)
    expected_scaler_count = len(splits) * len(seeds) if args.normalize_vol_targets and seeds else None

    report: Dict[str, Any] = {"status": "PASS", "warnings": warnings}
    report["events_targets"] = audit_events_targets(events, targets, warnings, args.num_levels)
    report["splits"] = audit_splits(events, splits, warnings)
    report["targets"] = audit_targets(
        targets,
        events,
        scaler_dir,
        warnings,
        normalize_vol_targets=args.normalize_vol_targets,
        expected_scaler_count=expected_scaler_count,
    )
    report["baselines"] = audit_baselines(events, splits, baseline_data_dir, aligned_summary, warnings)
    report["ridge_audit"] = audit_ridge_summary(ridge_audit_summary, warnings)
    report["masks"] = audit_masks(events, warnings, args.num_levels)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit temporal integrity, targets, splits, and baseline alignment.")
    parser.add_argument("--events", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-levels", type=int, default=10)
    parser.add_argument("--ctgnn-summary", default=None)
    parser.add_argument("--baseline-data-dir", default=None)
    parser.add_argument("--aligned-summary", default=None)
    parser.add_argument("--scaler-dir", default=None)
    parser.add_argument("--normalize-vol-targets", action="store_true")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--ridge-audit-summary", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    try:
        report = run_audit(args)
        write_reports(report, out_dir)
        print(f"AUDIT PASS | warnings={len(report['warnings'])} | report={out_dir / 'audit_report.json'}")
    except AssertionError as exc:
        report = {"status": "FAIL", "error": str(exc), "warnings": []}
        write_reports(report, out_dir)
        print(f"AUDIT FAIL | {exc}")
        raise


if __name__ == "__main__":
    main()
