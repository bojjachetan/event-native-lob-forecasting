from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, payload: Dict[str, object], indent: int = 2) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)


def format_mean_std(mean_val: float, std_val: float, decimals: int = 4) -> str:
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def summary_dict_to_frame(summary: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for metric, stats in summary.items():
        rows.append(
            {
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"],
                "formatted": stats["formatted"],
            }
        )
    return pd.DataFrame(rows)
