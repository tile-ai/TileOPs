from __future__ import annotations

import atexit
import contextlib
import gzip
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.cpp_extension import load

_EXT = None
_SESSION_LOCK = threading.RLock()
_SESSION: dict[str, Any] = {
    "active": False,
    "collector_active": False,
    "segments": [],
    "captures": [],
    "ops": [],
    "case_id": None,
    "current_op_index": None,
    "current_capture_index": None,
}
_CASE_COUNTER = 0
BENCHMARK_PROTOCOL = "native-cupti-sol-phase-isolated-v3"


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
    except RuntimeError as exc:
        raise NativeCUPTIError(f"Could not build/load native CUPTI collector: {exc}") from exc
    return _EXT


def case_session_active() -> bool:
    return bool(_SESSION["active"])


def _start_collector() -> None:
    if _SESSION["collector_active"]:
        raise RuntimeError("native CUPTI collector is already active")
    _call_extension(load_extension().start)
    _SESSION["collector_active"] = True


def _stop_collector(*, capture_phase: str | None = None) -> dict[str, Any]:
    if not _SESSION["collector_active"]:
        return {"kernels": [], "dropped": 0}
    ext = load_extension()
    try:
        torch.cuda.synchronize()
        _call_extension(ext.stop)
        trace = dict(_call_extension(ext.results))
        if capture_phase is not None:
            trace["capture_phase"] = capture_phase
            _SESSION["captures"].append(trace)
        return trace
    finally:
        _SESSION["collector_active"] = False


@contextlib.contextmanager
def _phase_session(phase: str):
    """Own one collector session for discovery or timing."""
    _start_collector()
    _SESSION["current_capture_index"] = len(_SESSION["captures"])
    try:
        yield
    finally:
        try:
            _stop_collector(capture_phase=phase)
        finally:
            _SESSION["current_capture_index"] = None


def start_case_session(case_id: str | None = None) -> None:
    """Start the attribution ledger owned by a pytest benchmark case."""
    with _SESSION_LOCK:
        if _SESSION["active"]:
            raise RuntimeError("native CUPTI case session is already active")
        # Extension build and CUDA initialization are intentionally outside
        # every phase-scoped activity capture.
        load_extension()
        torch.cuda.synchronize()
        _SESSION["segments"] = []
        _SESSION["captures"] = []
        _SESSION["ops"] = []
        _SESSION["case_id"] = case_id
        _SESSION["current_op_index"] = None
        _SESSION["current_capture_index"] = None
        _SESSION["active"] = True


def _checkpoint() -> dict[str, int]:
    if not _SESSION["active"]:
        start_case_session("standalone")
    if not _SESSION["collector_active"]:
        raise RuntimeError("native CUPTI checkpoint requested outside a collector session")
    raw = _call_extension(load_extension().checkpoint)
    return {
        "kernel_index": int(raw["kernel_index"]),
        "dropped": int(raw["dropped"]),
    }


def _trace_between(
    begin: dict[str, int],
    end: dict[str, int],
    *,
    phase: str,
    repeats: int,
) -> dict[str, Any]:
    raw = _call_extension(
        load_extension().results_range,
        begin["kernel_index"],
        end["kernel_index"],
        begin["dropped"],
    )
    trace = {
        "kernels": list(raw["kernels"]),
        "dropped": int(raw["dropped"]),
        "begin_checkpoint": begin,
        "end_checkpoint": end,
        "phase": phase,
        "repeats": repeats,
    }
    _SESSION["segments"].append({
        "op_index": _SESSION["current_op_index"],
        "capture_index": _SESSION.get("current_capture_index"),
        "phase": phase,
        "repeats": repeats,
        "begin_kernel_index": begin["kernel_index"],
        "end_kernel_index": end["kernel_index"],
        "dropped": trace["dropped"],
    })
    return trace


def begin_op() -> int:
    """Open an independently attributed op segment inside the current case."""
    if not _SESSION["active"]:
        start_case_session("standalone")
    if _SESSION["current_op_index"] is not None:
        raise RuntimeError("nested bench_kernel calls are not supported")
    op_index = len(_SESSION["ops"]) + 1
    _SESSION["current_op_index"] = op_index
    _SESSION["ops"].append({
        "op_index": op_index,
        "status": "running",
        "segment_begin": len(_SESSION["segments"]),
    })
    return op_index


def finish_op(op_index: int, *, status: str, error: str | None = None) -> None:
    """Close an op ledger entry without ending the enclosing case session."""
    if _SESSION["current_op_index"] != op_index:
        raise RuntimeError("native CUPTI op ledger is out of order")
    entry = _SESSION["ops"][op_index - 1]
    entry["status"] = status
    entry["segment_end"] = len(_SESSION["segments"])
    if error:
        entry["error"] = error
    _SESSION["current_op_index"] = None


def collect_discovery(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Capture prepare and operator activity separately during untimed discovery.

    Checkpoints are deliberately per phase here: discovery is not a latency
    sample.  Timed trials use only two checkpoints around the complete trial.
    """
    prepare_traces = []
    operator_traces = []
    with _phase_session("discovery"):
        for i in range(n_repeat):
            begin_prepare = _checkpoint()
            prepare_one(i)
            end_prepare = _checkpoint()
            prepare_traces.append(
                _trace_between(begin_prepare, end_prepare, phase="discovery_prepare", repeats=1)
            )

            begin_operator = end_prepare
            run_one(i)
            torch.cuda.synchronize()
            end_operator = _checkpoint()
            operator_traces.append(
                _trace_between(begin_operator, end_operator, phase="discovery_operator", repeats=1)
            )
    return prepare_traces, operator_traces


def collect_repeats(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Capture a complete timed trial as one ordered activity-record range."""
    with _phase_session("timing"):
        begin = _checkpoint()
        for i in range(n_repeat):
            if prepare_one is not None:
                prepare_one(i)
            run_one(i)
            torch.cuda.synchronize()
        end = _checkpoint()
        return _trace_between(begin, end, phase="timing", repeats=n_repeat)


def stop_case_session() -> None:
    """Close and optionally persist the case attribution ledger and captures."""
    global _CASE_COUNTER
    with _SESSION_LOCK:
        if not _SESSION["active"]:
            return
        try:
            torch.cuda.synchronize()
            captures = list(_SESSION["captures"])
            trace = {
                "dropped": sum(int(capture.get("dropped", 0)) for capture in captures),
                "captures": captures,
            }
            _CASE_COUNTER += 1
            trace["case_window"] = {
                "benchmark_file": os.environ.get("TILEOPS_CUPTI_BENCH_FILE"),
                "case_id": _SESSION["case_id"],
                "case_index": _CASE_COUNTER,
                "session_scope": "phase",
            }
            trace["segments"] = list(_SESSION["segments"])
            trace["ops"] = list(_SESSION["ops"])
            trace_dir = os.environ.get("TILEOPS_CUPTI_CASE_TRACE_DIR")
            if trace_dir:
                path = Path(
                    Path(trace_dir)
                    / f"cupti_case_{os.getpid()}_{_CASE_COUNTER:05d}.json.gz"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(
                    path,
                    "wt",
                    encoding="utf-8",
                    compresslevel=1,
                ) as stream:
                    json.dump(trace, stream, separators=(",", ":"), default=str)
        finally:
            if _SESSION["collector_active"]:
                _stop_collector()
            _SESSION["active"] = False
            _SESSION["current_op_index"] = None


atexit.register(stop_case_session)
