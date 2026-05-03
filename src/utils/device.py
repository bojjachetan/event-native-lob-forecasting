from __future__ import annotations

from typing import Dict

import torch


def _mps_built() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def device_backend_status() -> Dict[str, bool]:
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_built": _mps_built(),
        "mps_available": _mps_available(),
    }


def resolve_device(requested: str | None) -> str:
    req = (requested or "cpu").strip().lower()
    status = device_backend_status()

    if req in {"", "auto"}:
        if status["cuda_available"]:
            return "cuda"
        # For this TGN/PyG workload, Apple MPS benchmarked substantially slower
        # than CPU on local M3 Max hardware. Keep MPS available only when the
        # user explicitly requests --device mps.
        return "cpu"

    if req == "cuda":
        if not status["cuda_available"]:
            raise ValueError("Requested device=cuda but CUDA is not available in this environment.")
        return "cuda"

    if req == "mps":
        if not status["mps_available"]:
            raise ValueError(
                "Requested device=mps but Apple Metal is not available. "
                f"mps_built={status['mps_built']} mps_available={status['mps_available']}."
            )
        return "mps"

    if req == "cpu":
        return "cpu"

    raise ValueError("device must be one of: auto, cpu, cuda, mps")


def describe_device(resolved_device: str) -> str:
    status = device_backend_status()
    return (
        f"Resolved device: {resolved_device} | "
        f"cuda_available={status['cuda_available']} | "
        f"mps_built={status['mps_built']} | "
        f"mps_available={status['mps_available']}"
    )
