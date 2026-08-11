"""Native CUPTI activity collection for SOL-style benchmark timing."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.cpp_extension import load

_EXT = None
_COLLECTOR_ACTIVE = False


class NativeCUPTIError(RuntimeError):
    """The native CUPTI collector could not be built or operated."""


def _call_extension(function: Callable, *args):
    try:
        return function(*args)
    except RuntimeError as exc:
        raise NativeCUPTIError(f"Native CUPTI collector failed: {exc}") from exc


def load_extension():
    """Build and load the tiny native CUPTI activity collector."""
    global _EXT
    if _EXT is not None:
        return _EXT

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    source = Path(__file__).with_name("native_cupti.cpp")

    cuda_include_candidates = [
        cuda_home / "targets" / "x86_64-linux" / "include",
        Path("/usr/local/cuda/targets/x86_64-linux/include"),
    ]
    cupti_include_candidates = [
        cuda_home / "extras" / "CUPTI" / "include",
        cuda_home / "targets" / "x86_64-linux" / "include",
        Path("/usr/local/cuda/extras/CUPTI/include"),
        Path("/usr/local/cuda/targets/x86_64-linux/include"),
    ]
    lib_candidates = [
        cuda_home / "targets" / "x86_64-linux" / "lib",
        cuda_home / "extras" / "CUPTI" / "lib64",
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda/extras/CUPTI/lib64"),
    ]

    cuda_include_dir = next(
        (p for p in cuda_include_candidates if (p / "cuda_runtime_api.h").exists()),
        None,
    )
    cupti_include_dir = next(
        (p for p in cupti_include_candidates if (p / "cupti.h").exists()),
        None,
    )
    lib_dir = next((p for p in lib_candidates if (p / "libcupti.so").exists()), None)
    if cuda_include_dir is None or cupti_include_dir is None or lib_dir is None:
        raise NativeCUPTIError(
            "Could not locate CUPTI headers/library. Set CUDA_HOME to a CUDA "
            "toolkit path that contains CUPTI."
        )
    include_dirs = list(dict.fromkeys([str(cuda_include_dir), str(cupti_include_dir)]))

    try:
        _EXT = load(
            name="tileops_native_cupti_ext",
            sources=[str(source)],
            extra_include_paths=include_dirs,
            extra_cflags=["-O2", "-std=c++17"],
            extra_ldflags=[f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}", "-lcupti"],
            verbose=bool(int(os.environ.get("TILEOPS_NATIVE_CUPTI_VERBOSE", "0"))),
        )
    except Exception as exc:
        # A failed build surfaces as RuntimeError, but importing the result can
        # fail with ImportError/OSError; both must reach the same fallback.
        raise NativeCUPTIError(f"Could not build/load native CUPTI collector: {exc}") from exc
    return _EXT


@contextlib.contextmanager
def _phase_session():
    """Own one collector session for discovery or timing.

    Discovery and timing never share a session: a sequence mismatch found
    during discovery must not leave records that timing would attribute.
    """
    global _COLLECTOR_ACTIVE
    if _COLLECTOR_ACTIVE:
        raise RuntimeError("native CUPTI collector is already active")
    _call_extension(load_extension().start)
    _COLLECTOR_ACTIVE = True
    try:
        yield
    finally:
        try:
            _call_extension(load_extension().stop)
        finally:
            _COLLECTOR_ACTIVE = False


def _checkpoint() -> dict[str, int]:
    if not _COLLECTOR_ACTIVE:
        raise RuntimeError("native CUPTI checkpoint requested outside a collector session")
    raw = _call_extension(load_extension().checkpoint)
    return {"kernel_index": int(raw["kernel_index"]), "dropped": int(raw["dropped"])}


def _trace_between(begin: dict[str, int], end: dict[str, int]) -> dict[str, Any]:
    raw = _call_extension(
        load_extension().results_range,
        begin["kernel_index"],
        end["kernel_index"],
        begin["dropped"],
    )
    return {"kernels": list(raw["kernels"]), "dropped": int(raw["dropped"])}


def collect_discovery(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Capture prepare and operator activity separately during untimed discovery.

    Checkpoints are per phase here because discovery is not a latency sample.
    Timed repeats use two checkpoints around the whole trial instead.
    """
    prepare_traces = []
    operator_traces = []
    with _phase_session():
        for i in range(n_repeat):
            begin_prepare = _checkpoint()
            prepare_one(i)
            end_prepare = _checkpoint()
            prepare_traces.append(_trace_between(begin_prepare, end_prepare))

            run_one(i)
            operator_traces.append(_trace_between(end_prepare, _checkpoint()))
    return prepare_traces, operator_traces


def collect_repeats(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Capture a complete timed trial as one ordered activity-record range."""
    with _phase_session():
        begin = _checkpoint()
        for i in range(n_repeat):
            if prepare_one is not None:
                prepare_one(i)
            run_one(i)
            torch.cuda.synchronize()
        return _trace_between(begin, _checkpoint())
