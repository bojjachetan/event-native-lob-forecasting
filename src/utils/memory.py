from __future__ import annotations

import os
import resource
from typing import Iterable, Mapping, Sequence, Tuple


def get_process_memory_gb() -> float:
    """Return current process resident memory in GiB when available."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024**3)
    except Exception:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KiB.
        if usage > 10**9:
            return float(usage) / (1024**3)
        return float(usage) / (1024**2)


def estimate_tensor_memory_gb(shapes: Mapping[str, Sequence[int]] | Iterable[Tuple[str, Sequence[int]]], dtype_bytes: int = 4) -> float:
    """Estimate dense tensor memory for a collection of named shapes."""
    items = shapes.items() if isinstance(shapes, Mapping) else shapes
    total = 0
    for _, shape in items:
        n = 1
        for dim in shape:
            n *= int(dim)
        total += n * dtype_bytes
    return float(total) / (1024**3)


def log_memory(stage_name: str) -> float:
    """Print and return process memory in GiB for lightweight runtime tracing."""
    gb = get_process_memory_gb()
    print(f"[memory] {stage_name}: {gb:.3f} GiB")
    return gb
