from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.device import describe_device, resolve_device
from src.utils.memory import get_process_memory_gb


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Namespace):
        return {k: _jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def get_git_commit(repo_root: str | Path = ".") -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def get_git_dirty(repo_root: str | Path = ".") -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return None


def collect_environment(device: str = "cpu", repo_root: str | Path = ".") -> Dict[str, Any]:
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore

    try:
        import torch_geometric  # type: ignore

        pyg_version = torch_geometric.__version__
    except Exception:
        pyg_version = None

    resolved_device = resolve_device(device)
    device_info: Dict[str, Any] = {
        "requested": device,
        "resolved": resolved_device,
        "description": describe_device(resolved_device),
    }

    if torch is not None:
        device_info.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
                "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            }
        )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "pyg_version": pyg_version,
        "device": device_info,
        "git_commit": get_git_commit(repo_root),
        "git_dirty": get_git_dirty(repo_root),
        "process_memory_gb": get_process_memory_gb(),
    }


def build_run_manifest(
    *,
    args: Any = None,
    config: Any = None,
    repo_root: str | Path = ".",
    device: str = "cpu",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "environment": collect_environment(device=device, repo_root=repo_root),
        "command_line_args": _jsonable(args) if args is not None else None,
        "resolved_config": _jsonable(config) if config is not None else None,
        "metadata": _jsonable(metadata or {}),
    }
    return manifest


def save_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)


def update_run_manifest(path: str | Path, updates: Mapping[str, Any]) -> Dict[str, Any]:
    out = Path(path)
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest.update(_jsonable(updates))
    save_run_manifest(out, manifest)
    return manifest
