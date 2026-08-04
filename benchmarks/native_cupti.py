from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from torch.utils.cpp_extension import load


_EXT = None


def load_extension():
    """Build and load the tiny native CUPTI activity collector."""
    global _EXT
    if _EXT is not None:
        return _EXT

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    source = Path(__file__).with_name("native_cupti.cpp")

    include_candidates = [
        cuda_home / "targets" / "x86_64-linux" / "include",
        cuda_home / "extras" / "CUPTI" / "include",
        Path("/usr/local/cuda/targets/x86_64-linux/include"),
        Path("/usr/local/cuda/extras/CUPTI/include"),
    ]
    lib_candidates = [
        cuda_home / "targets" / "x86_64-linux" / "lib",
        cuda_home / "extras" / "CUPTI" / "lib64",
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda/extras/CUPTI/lib64"),
    ]

    include_dir = next((p for p in include_candidates if (p / "cupti.h").exists()), None)
    lib_dir = next((p for p in lib_candidates if p.exists()), None)
    if include_dir is None or lib_dir is None:
        raise RuntimeError(
            "Could not locate CUPTI headers/library. Set CUDA_HOME to a CUDA "
            "toolkit path that contains CUPTI."
        )

    _EXT = load(
        name="tileops_native_cupti_ext",
        sources=[str(source)],
        extra_include_paths=[str(include_dir)],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=[f"-L{lib_dir}", "-lcupti"],
        verbose=bool(int(os.environ.get("TILEOPS_NATIVE_CUPTI_VERBOSE", "0"))),
    )
    return _EXT


def collect_repeats(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Collect CUPTI activities for ``n_repeat`` logical calls.

    ``run_one(i)`` is called once per repeat. Each call is wrapped by a CPU
    timestamp window taken from ``cuptiGetTimestamp``. CUDA work is synchronized
    before the end timestamp so every kernel activity for the call should fall
    inside that repeat's attribution window.
    """
    ext = load_extension()
    started = False
    try:
        ext.start()
        started = True
        for i in range(n_repeat):
            if prepare_one is not None:
                prepare_one(i)
            ext.begin_repeat(i)
            try:
                run_one(i)
                # Host-side wait only; the selected latency comes from CUPTI
                # GPU activity timestamps.
                import torch

                torch.cuda.synchronize()
            finally:
                ext.end_repeat(i)
    finally:
        if started:
            ext.stop()
    return ext.results()
