import contextlib
import json
import logging
import os
import random
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
)

import pytest
import torch

from tileops.manifest import (
    WORKLOAD_RESERVED_KEYS,
    load_manifest,
    load_workloads,
    single_input_workload_contract,
)

from . import native_cupti


def _workload_contract(op_name: str) -> tuple[str, frozenset[str]]:
    """Resolve the shared workload contract for an op known to exist."""
    sig = load_manifest()[op_name].get("signature") or {}
    contract = single_input_workload_contract(sig)
    if contract is None:
        raise KeyError(
            f"workloads_to_params({op_name!r}) needs exactly one manifest "
            "tensor input; multi-input ops use their own bench files."
        )
    return contract


W = TypeVar("W")


_logger = logging.getLogger("tileops.bench")

# Thread-local storage for conftest hook to pick up per-test bench results.
# A single test function may call record() multiple times (tileops + baseline).
_bench_results = threading.local()

# Latest bench_kernel measurement metadata; deviations from the default
# protocol are surfaced in results by BenchmarkBase._build_result.
_bench_meta = threading.local()
_trace_dump_counter = 0
_cuda_runtime = None


class _NativeCuptiAttributionError(Exception):
    """Native CUPTI trace could not be attributed to logical benchmark calls."""


class _ShiftingTensorPool:
    """SOL-style input pool returning a unique tensor data_ptr per call."""

    _POOL_ALIGNMENT = 256
    _MAX_SHIFT_BYTES = 2048

    def __init__(
        self,
        args: tuple[Any, ...],
        total_iterations: int,
        *,
        seed: int,
    ) -> None:
        self._call_idx = 0
        self._total_iterations = total_iterations
        self._offset_blocks = self._compute_offset_blocks(total_iterations, seed)
        self._entries = [self._make_entry(arg) for arg in args]

    @classmethod
    def _compute_offset_blocks(cls, total_iterations: int, seed: int) -> list[int]:
        max_multiplier = cls._MAX_SHIFT_BYTES // cls._POOL_ALIGNMENT
        rng = random.Random(seed)
        offsets = [0]
        for _ in range(max(0, total_iterations - 1)):
            offsets.append(offsets[-1] + rng.randint(1, max_multiplier))
        return offsets

    @staticmethod
    def _storage_span(tensor: torch.Tensor) -> int:
        if tensor.numel() == 0:
            return 0
        span = 1
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
            if size > 1:
                span += (size - 1) * stride
        return span

    def _make_entry(self, arg: Any) -> dict[str, Any]:
        if not isinstance(arg, torch.Tensor):
            return {"scalar": arg}
        tensor = arg.contiguous() if any(stride < 0 for stride in arg.stride()) else arg
        storage_span = self._storage_span(tensor)
        elem_size = tensor.element_size()
        block_numel = max(1, self._POOL_ALIGNMENT // elem_size)
        pool_numel = storage_span + self._offset_blocks[-1] * block_numel
        pool = torch.empty(pool_numel, dtype=tensor.dtype, device=tensor.device)
        source = tensor.as_strided((storage_span,), (1,))
        return {
            "pool": pool,
            "source": source,
            "shape": tuple(tensor.shape),
            "strides": tensor.stride(),
            "storage_span": storage_span,
            "block_numel": block_numel,
        }

    def next_args(self) -> tuple[Any, ...]:
        if self._call_idx >= self._total_iterations:
            raise RuntimeError(
                "ShiftingTensorPool exhausted: called "
                f"{self._call_idx + 1} times but allocated for "
                f"{self._total_iterations} iterations"
            )

        block_offset = self._offset_blocks[self._call_idx]
        result: list[Any] = []
        for entry in self._entries:
            if "scalar" in entry:
                result.append(entry["scalar"])
                continue
            start = block_offset * entry["block_numel"]
            entry["pool"].narrow(0, start, entry["storage_span"]).copy_(entry["source"])
            result.append(
                entry["pool"].as_strided(entry["shape"], entry["strides"], start)
            )
        self._call_idx += 1
        return tuple(result)


def _kernels_by_cpu_window(trace: dict) -> list[list[dict]]:
    """Group CUPTI kernel activities by per-repeat CPU attribution windows."""
    windows = _valid_cpu_windows(trace)
    kernels = sorted(
        trace.get("kernels", []),
        key=lambda k: (int(k["start_ns"]), int(k["end_ns"])),
    )
    begin_tolerance_ns = _cupti_window_begin_tolerance_ns()
    end_tolerance_ns = _cupti_window_end_tolerance_ns()
    grouped: list[list[dict]] = []
    for _, begin_ns, end_ns in windows:
        grouped.append([
            k for k in kernels
            if begin_ns - begin_tolerance_ns <= int(k["start_ns"])
            and int(k["end_ns"]) <= end_ns + end_tolerance_ns
        ])
    return grouped


def _cupti_window_begin_tolerance_ns() -> int:
    value_us = float(os.getenv("TILEOPS_CUPTI_WINDOW_BEGIN_TOLERANCE_US", "2.0"))
    return int(value_us * 1000)


def _cupti_window_end_tolerance_ns() -> int:
    # CUPTI CPU and activity timestamps can disagree by a few microseconds at
    # the tail of very short multi-kernel calls. Keep this below the repeat
    # guard so the widened attribution window cannot reach the next prepare.
    value_us = float(os.getenv("TILEOPS_CUPTI_WINDOW_END_TOLERANCE_US", "8.0"))
    return int(value_us * 1000)


def _valid_cpu_windows(trace: dict) -> list[tuple[int, int, int]]:
    windows = sorted(
        (
            (int(w["repeat"]), int(w["begin_ns"]), int(w["end_ns"]))
            for w in trace.get("cpu_windows", [])
            if int(w.get("begin_ns", 0)) > 0 and int(w.get("end_ns", 0)) > 0
        ),
        key=lambda item: item[0],
    )
    return windows


def _kernel_sequence(kernels: list[dict]) -> tuple[str, ...]:
    return tuple(str(k["name"]) for k in kernels)


def _select_expected_sequence(
    kernels: list[dict],
    expected_sequence: tuple[str, ...],
) -> list[dict] | None:
    """Select a complete discovered kernel sequence from one logical call.

    Every kernel activity attributed to the call must belong to the discovered
    sequence. Silently discarding an unknown activity could underestimate the
    call when a dynamic path launches an extra kernel before or after the
    expected sequence.
    """
    if not expected_sequence:
        return None

    expected_count = len(expected_sequence)
    if len(kernels) != expected_count:
        return None

    candidates = kernels
    candidate_names = _kernel_sequence(candidates)

    for start_idx in range(len(candidates) - expected_count + 1):
        end_idx = start_idx + expected_count
        if candidate_names[start_idx:end_idx] == expected_sequence:
            return candidates[start_idx:end_idx]

    expected_counts = Counter(expected_sequence)
    best: list[dict] | None = None
    best_score: tuple[int, float] | None = None
    for start_idx in range(len(candidates) - expected_count + 1):
        end_idx = start_idx + expected_count
        names = candidate_names[start_idx:end_idx]
        if Counter(names) != expected_counts:
            continue
        selected = candidates[start_idx:end_idx]
        score = (
            _relative_order_score(names, expected_sequence),
            -_kernel_span_us(selected),
        )
        if best_score is None or score > best_score:
            best_score = score
            best = selected

    return best


def _relative_order_score(
    actual: tuple[str, ...],
    expected: tuple[str, ...],
) -> int:
    """Longest-common-subsequence score for deterministic tie breaking."""
    previous = [0] * (len(expected) + 1)
    for actual_name in actual:
        current = [0] * (len(expected) + 1)
        for idx, expected_name in enumerate(expected, start=1):
            if actual_name == expected_name:
                current[idx] = previous[idx - 1] + 1
            else:
                current[idx] = max(previous[idx], current[idx - 1])
        previous = current
    return previous[-1]


def _kernel_span_us(kernels: list[dict]) -> float:
    if not kernels:
        return 0.0
    start_ns = min(int(kernel["start_ns"]) for kernel in kernels)
    end_ns = max(int(kernel["end_ns"]) for kernel in kernels)
    return (end_ns - start_ns) / 1000.0


def _short_kernel_name(name: str) -> str:
    if len(name) <= 96:
        return name
    return name[:93] + "..."


def _format_sequence(seq: tuple[str, ...]) -> str:
    if not seq:
        return "<empty>"
    names = [_short_kernel_name(name) for name in seq]
    if len(names) <= 4:
        return " -> ".join(names)
    return " -> ".join(names[:2] + ["..."] + names[-2:])


def _trace_diagnostic(
    trace: dict,
    *,
    expected_sequence: tuple[str, ...] | None = None,
    max_examples: int = 4,
) -> str:
    grouped = _kernels_by_cpu_window(trace)
    sequences = [_kernel_sequence(kernels) for kernels in grouped]
    sequence_counts = Counter(sequences)
    kernels = sorted(
        trace.get("kernels", []),
        key=lambda k: (int(k["start_ns"]), int(k["end_ns"])),
    )
    windows = _valid_cpu_windows(trace)

    parts = [
        f"windows={len(trace.get('cpu_windows', []))}",
        f"grouped_windows={len(grouped)}",
        f"kernels={len(kernels)}",
        f"dropped={int(trace.get('dropped', 0))}",
        f"begin_tolerance_us={_cupti_window_begin_tolerance_ns() / 1000.0:.3f}",
        f"end_tolerance_us={_cupti_window_end_tolerance_ns() / 1000.0:.3f}",
    ]
    if expected_sequence is not None:
        selected = [
            _select_expected_sequence(kernels, expected_sequence)
            for kernels in grouped
        ]
        matched = sum(1 for kernels in selected if kernels is not None)
        parts.append(f"matched={matched}/{len(grouped)}")
        parts.append(f"expected=[{_format_sequence(expected_sequence)}]")

    common = []
    for seq, count in sequence_counts.most_common(max_examples):
        common.append(f"{count}x[{_format_sequence(seq)}]")
    if common:
        parts.append("sequences=" + "; ".join(common))

    examples = []
    for idx, kernels_in_window in enumerate(grouped[:max_examples]):
        seq = _kernel_sequence(kernels_in_window)
        examples.append(f"r{idx}:len={len(seq)}[{_format_sequence(seq)}]")
    if examples:
        parts.append("examples=" + "; ".join(examples))

    unmatched = []
    for idx, (repeat, begin_ns, end_ns) in enumerate(windows):
        if idx >= len(sequences):
            continue
        if expected_sequence is None:
            bad = idx > 0 and sequences[idx] != sequences[0]
        else:
            bad = _select_expected_sequence(grouped[idx], expected_sequence) is None
        if not bad:
            continue

        span_us = (end_ns - begin_ns) / 1000.0
        nearby = sorted(
            kernels,
            key=lambda k: _distance_to_window_us(k, begin_ns, end_ns),
        )[:3]
        nearby_text = []
        for k in nearby:
            start_delta = (int(k["start_ns"]) - begin_ns) / 1000.0
            end_delta = (int(k["end_ns"]) - end_ns) / 1000.0
            nearby_text.append(
                f"{_short_kernel_name(str(k['name']))}"
                f"(start-begin={start_delta:.1f}us,end-end={end_delta:.1f}us)"
            )
        unmatched.append(
            f"r{repeat}:span={span_us:.1f}us seq=[{_format_sequence(sequences[idx])}] "
            f"near=" + ",".join(nearby_text)
        )
        if len(unmatched) >= max_examples:
            break
    if unmatched:
        parts.append("unmatched=" + "; ".join(unmatched))
    return "; ".join(parts)


def _trace_window_analysis(
    trace: dict,
    *,
    expected_sequence: tuple[str, ...] | None = None,
    nearest: int = 5,
) -> list[dict]:
    grouped = _kernels_by_cpu_window(trace)
    sequences = [_kernel_sequence(kernels) for kernels in grouped]
    kernels = sorted(
        trace.get("kernels", []),
        key=lambda k: (int(k["start_ns"]), int(k["end_ns"])),
    )

    rows: list[dict] = []
    for idx, (repeat, begin_ns, end_ns) in enumerate(_valid_cpu_windows(trace)):
        seq = sequences[idx] if idx < len(sequences) else ()
        selected = (
            _select_expected_sequence(grouped[idx], expected_sequence)
            if expected_sequence is not None and idx < len(grouped)
            else None
        )
        matched = (
            selected is not None
            if expected_sequence is not None
            else idx == 0 or seq == sequences[0]
        )
        near = []
        for k in sorted(
            kernels,
            key=lambda item: _distance_to_window_us(item, begin_ns, end_ns),
        )[:nearest]:
            near.append({
                "name": str(k["name"]),
                "start_ns": int(k["start_ns"]),
                "end_ns": int(k["end_ns"]),
                "start_minus_begin_us": (int(k["start_ns"]) - begin_ns) / 1000.0,
                "end_minus_end_us": (int(k["end_ns"]) - end_ns) / 1000.0,
                "contained": begin_ns - _cupti_window_begin_tolerance_ns() <= int(k["start_ns"])
                and int(k["end_ns"]) <= end_ns + _cupti_window_end_tolerance_ns(),
                "distance_to_window_us": _distance_to_window_us(k, begin_ns, end_ns),
            })
        rows.append({
            "repeat": repeat,
            "begin_ns": begin_ns,
            "end_ns": end_ns,
            "span_us": (end_ns - begin_ns) / 1000.0,
            "sequence": list(seq),
            "selected_sequence": (
                list(_kernel_sequence(selected)) if selected is not None else None
            ),
            "matched": matched,
            "nearest_kernels": near,
        })
    return rows


def _dump_trace(
    phase: str,
    trace: dict,
    *,
    expected_sequence: tuple[str, ...] | None = None,
    reason: str,
) -> str | None:
    dump_dir = os.getenv("TILEOPS_CUPTI_TRACE_DUMP_DIR")
    if not dump_dir:
        return None

    global _trace_dump_counter
    _trace_dump_counter += 1
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = path / f"cupti_trace_{os.getpid()}_{_trace_dump_counter:04d}_{phase}_{stamp}.json"
    payload = {
        "phase": phase,
        "reason": reason,
        "expected_sequence": list(expected_sequence) if expected_sequence is not None else None,
        "diagnostic": _trace_diagnostic(trace, expected_sequence=expected_sequence),
        "window_analysis": _trace_window_analysis(
            trace,
            expected_sequence=expected_sequence,
        ),
        "trace": trace,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    return str(out)


def _distance_to_window_us(kernel: dict, begin_ns: int, end_ns: int) -> float:
    start_ns = int(kernel["start_ns"])
    kernel_end_ns = int(kernel["end_ns"])
    if begin_ns <= start_ns and kernel_end_ns <= end_ns:
        return 0.0
    if kernel_end_ns < begin_ns:
        return (begin_ns - kernel_end_ns) / 1000.0
    if start_ns > end_ns:
        return (start_ns - end_ns) / 1000.0
    return min(abs(start_ns - begin_ns), abs(kernel_end_ns - end_ns)) / 1000.0


def _discover_expected_sequence(
    run_one: Callable[[int], None],
    prepare_one: Callable[[int], None],
) -> tuple[str, ...]:
    n_discovery = int(os.getenv("TILEOPS_CUPTI_DISCOVERY_REPEATS", "3"))
    trace = native_cupti.collect_repeats(run_one, n_discovery, prepare_one=prepare_one)
    if int(trace.get("dropped", 0)) != 0:
        raise _NativeCuptiAttributionError(
            f"CUPTI dropped {trace['dropped']} records during discovery"
        )

    sequences = [_kernel_sequence(kernels) for kernels in _kernels_by_cpu_window(trace)]
    if not sequences or any(len(seq) == 0 for seq in sequences):
        reason = "CUPTI discovery found no CUDA kernel sequence"
        dumped = _dump_trace("discovery", trace, reason=reason)
        suffix = f"; trace_dump={dumped}" if dumped else ""
        raise _NativeCuptiAttributionError(
            reason + "; " + _trace_diagnostic(trace) + suffix
        )
    first = sequences[0]
    if any(seq != first for seq in sequences[1:]):
        reason = "CUPTI discovery saw inconsistent kernel sequences across repeats"
        dumped = _dump_trace("discovery", trace, reason=reason)
        suffix = f"; trace_dump={dumped}" if dumped else ""
        raise _NativeCuptiAttributionError(
            reason + "; " + _trace_diagnostic(trace) + suffix
        )
    return first


def _attributed_mean_latency_ms(
    trace: dict,
    expected_sequence: tuple[str, ...],
    n_repeat: int,
) -> float:
    if int(trace.get("dropped", 0)) != 0:
        raise _NativeCuptiAttributionError(
            f"CUPTI dropped {trace['dropped']} records during timing"
        )

    samples_us: list[float] = []
    grouped = _kernels_by_cpu_window(trace)
    for kernels in grouped:
        selected = _select_expected_sequence(kernels, expected_sequence)
        if selected is None:
            continue
        samples_us.append(_kernel_span_us(selected))

    if not samples_us:
        reason = "CUPTI timing found no complete expected kernel sequence"
        dumped = _dump_trace(
            "timing",
            trace,
            expected_sequence=expected_sequence,
            reason=reason,
        )
        suffix = f"; trace_dump={dumped}" if dumped else ""
        raise _NativeCuptiAttributionError(
            reason + "; "
            + _trace_diagnostic(trace, expected_sequence=expected_sequence)
            + suffix
        )
    if len(samples_us) != n_repeat:
        reason = (
            f"CUPTI timing attributed {len(samples_us)}/{n_repeat} complete "
            "expected kernel sequences"
        )
        dumped = _dump_trace(
            "timing",
            trace,
            expected_sequence=expected_sequence,
            reason=reason,
        )
        suffix = f"; trace_dump={dumped}" if dumped else ""
        raise _NativeCuptiAttributionError(
            reason + "; "
            + _trace_diagnostic(trace, expected_sequence=expected_sequence)
            + suffix
        )

    _bench_meta.cupti_sampled_calls = len(samples_us)
    _bench_meta.cupti_expected_kernel_count = len(expected_sequence)
    return (sum(samples_us) / len(samples_us)) * 1e-3


# L2 cache flush buffer, allocated lazily.

_l2_flush_cache: Optional[torch.Tensor] = None


def _reset_persisting_l2_cache() -> None:
    global _cuda_runtime
    if _cuda_runtime is None:
        from cuda.bindings import runtime as cuda_runtime

        _cuda_runtime = cuda_runtime

    result = _cuda_runtime.cudaCtxResetPersistingL2Cache()
    if isinstance(result, tuple):
        result = result[0]
    torch.cuda.check_error(int(result))


def _get_l2_flush_cache() -> torch.Tensor:
    global _l2_flush_cache
    if _l2_flush_cache is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if l2_bytes <= 0:
            _logger.warning(
                "L2 cache size query returned %d; flushing a 256 MB buffer "
                "instead", l2_bytes,
            )
            l2_bytes = int(256e6)
        _l2_flush_cache = torch.empty(2 * l2_bytes, dtype=torch.int8, device="cuda")
    return _l2_flush_cache


def _native_output_suppressor():
    """Return an fd-level output suppressor that is safe under pytest capture.

    tilelang's ``suppress_stdout_stderr`` dup2's ``/dev/null`` over
    ``sys.stdout.fileno()``; under pytest fd capture that fileno is the
    capture tmpfile and the redirect corrupts it (``EBADF`` on later reads).
    Suppress only when stdout/stderr are the process fds 1/2.
    """
    try:
        native = sys.stdout.fileno() == 1 and sys.stderr.fileno() == 2
    except (AttributeError, OSError, ValueError):
        # Streams without a real descriptor (io.StringIO, capsys) or with
        # fileno() unsupported: fd-level suppression is impossible.
        native = False
    if not native:
        return contextlib.nullcontext()
    from tilelang.profiler.bench import suppress_stdout_stderr
    return suppress_stdout_stderr()


# NVIDIA SOL-ExecBench–style benchmark


def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    n_warmup: int = 10,
    n_repeat: int = 50,
    n_trials: int = 3,
) -> float:
    """Benchmark a GPU callable with CUPTI activity timing.

    Protocol (adapted from NVIDIA SOL-ExecBench, arxiv.org/abs/2603.19173):
      1. Lock GPU clocks externally (nvidia-smi).
      2. Run *n_warmup* un-timed iterations with L2 flush.
      3. For each of *n_trials* trials, profile *n_repeat* iterations
         under CUPTI to get activity time without host launch overhead.
         Persisting L2 is reset and a 2x-L2 buffer is cleared before every
         iteration. Input tensors are prepared in a SOL-style shifting pool
         before L2 preparation so each measured call uses a different data
         pointer without warming the flushed cache.
      4. Report the median trial mean (robust to outlier trials).

    Uses native CUPTI activity timestamps for SOL-style attribution instead of
    PyTorch profiler/Kineto projection. A discovery pass records the expected
    per-call kernel sequence; timed repeats then measure each complete sequence
    from the earliest selected kernel start to the latest selected kernel end.
    This degenerates to pure kernel duration for single-kernel calls and
    preserves inter-kernel gaps for multi-kernel calls. Falls back to CUDA
    events only if native CUPTI is unavailable or cannot attribute any complete
    call. CUDA events fallback is opt-in via
    ``TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1``.

    Args:
        fn: Callable to benchmark.  If *args* is provided, called with
            tensors from the shifting input pool; otherwise called as ``fn()``.
        args: Tensor arguments to rotate each iteration.  Non-tensor
            values are passed through unchanged.
        n_warmup: Warmup iterations (default 10).
        n_repeat: Timed iterations per trial (default 50).
        n_trials: Independent trials (default 3).

    Returns:
        Kernel latency in **milliseconds**.
    """
    if not isinstance(args, tuple):
        raise TypeError(
            f"bench_kernel expects a tuple of args, got {type(args).__name__}. "
            "Check that gen_inputs() returns a tuple."
        )

    _bench_meta.timing = None
    _bench_meta.input_policy = None
    _bench_meta.input_policy_seed = None
    _bench_meta.cupti_begin_tolerance_us = _cupti_window_begin_tolerance_ns() / 1000.0
    _bench_meta.cupti_end_tolerance_us = _cupti_window_end_tolerance_ns() / 1000.0
    _bench_meta.cupti_repeat_guard_us = native_cupti.repeat_guard_us()
    _bench_meta.cupti_sampled_calls = None
    _bench_meta.cupti_expected_kernel_count = None
    _bench_meta.fallback_reason = None

    cache = _get_l2_flush_cache()
    has_args = len(args) > 0

    if has_args:
        discovery_repeats = int(os.getenv("TILEOPS_CUPTI_DISCOVERY_REPEATS", "3"))
        total_iterations = (
            1  # first call
            + n_warmup
            + discovery_repeats
            + n_trials * n_repeat
            + n_trials * n_repeat  # possible CUDA-events fallback
        )
        seed = int(os.getenv("TILEOPS_INPUT_POOL_SEED", "0"))
        arg_pool = _ShiftingTensorPool(args, total_iterations, seed=seed)
        _bench_meta.input_policy = "shifting-pool"
        _bench_meta.input_policy_seed = seed
        prepared_args: tuple[Any, ...] | None = None

        def _prepare_args(i):
            nonlocal prepared_args
            prepared_args = arg_pool.next_args()

        def _run(i):
            if prepared_args is None:
                raise RuntimeError("bench_kernel called _run before preparing args")
            return fn(*prepared_args)
    else:
        arg_pool = None
        _bench_meta.input_policy = "none"

        def _prepare_args(i):
            return None

        def _run(i):
            return fn()

    def _prepare_iteration(i):
        _prepare_args(i)
        _reset_persisting_l2_cache()
        cache.zero_()
        torch.cuda.synchronize()

    # First call and warmup are outside CUPTI timing. They absorb CUDA context
    # init, module/library lazy init, JIT/autotune, and allocator growth.
    _prepare_iteration(0)
    _run(0)
    torch.cuda.synchronize()

    for i in range(n_warmup):
        _prepare_iteration(i)
        _run(i)
    torch.cuda.synchronize()

    trial_means: list[float] = []
    try:
        with _native_output_suppressor():
            expected_sequence = _discover_expected_sequence(_run, _prepare_iteration)
            for _ in range(n_trials):
                trace = native_cupti.collect_repeats(
                    _run,
                    n_repeat,
                    prepare_one=_prepare_iteration,
                )
                trial_means.append(
                    _attributed_mean_latency_ms(trace, expected_sequence, n_repeat)
                )
        _bench_meta.timing = "native-cupti"
    except _NativeCuptiAttributionError as exc:
        # Check if cuda-events fallback is allowed
        allow_fallback = os.getenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0") == "1"

        if not allow_fallback:
            raise RuntimeError(
                f"Native CUPTI profiling failed: {exc}. "
                "CUDA-events fallback is disabled (TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0). "
                "This prevents silently mixing CUPTI timing with CUDA-events timing. "
                "To debug: run with TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1 and check logs."
            ) from exc

        _bench_meta.fallback_reason = str(exc)
        _bench_meta.cupti_sampled_calls = None
        _bench_meta.cupti_expected_kernel_count = None
        _logger.warning(
            "Native CUPTI attribution failed (%s); falling back to CUDA-events "
            "timing, which includes launch/stream gaps per call. "
            "Set TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0 to prevent fallback.", exc,
        )
        trial_means = []
    except RuntimeError as exc:
        if "CUPTI" not in str(exc):
            raise
        allow_fallback = os.getenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0") == "1"
        if not allow_fallback:
            raise
        _bench_meta.fallback_reason = str(exc)
        _bench_meta.cupti_sampled_calls = None
        _bench_meta.cupti_expected_kernel_count = None
        _logger.warning(
            "Native CUPTI setup failed (%s); falling back to CUDA-events timing.",
            exc,
        )
        trial_means = []

    # Fallback to CUDA events if CUPTI failed
    if not trial_means:
        _bench_meta.timing = "cuda-events"
        # Mimic CUPTI behavior: flush L2 before measurement window
        for _ in range(n_trials):
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

            for i in range(n_repeat):
                _prepare_iteration(i)
                start_events[i].record()
                _run(i)
                end_events[i].record()
            torch.cuda.synchronize()

            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events, strict=True)]
            trial_means.append(sum(times) / len(times))

    # Free the arg pool and release cached GPU memory to prevent
    # accumulation across hundreds of benchmark calls.
    if arg_pool is not None:
        del arg_pool
    torch.cuda.empty_cache()

    trial_means.sort()
    return trial_means[len(trial_means) // 2]


def _get_env_metadata() -> list[str]:
    """Collect GPU model, driver version, CUDA version, and torch version."""
    lines = []
    lines.append(f"- **Torch version**: {torch.__version__}")
    lines.append(f"- **CUDA version (torch)**: {torch.version.cuda or 'N/A'}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        lines.append(f"- **GPU model**: {gpu_name}")
    else:
        lines.append("- **GPU model**: N/A (no CUDA device)")

    # Try to get NVIDIA driver version and clocks from nvidia-smi.
    gpu_query_fields = [
        "driver_version",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.applications.graphics",
        "clocks.applications.memory",
    ]
    gpu_query_values = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(gpu_query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpu_query_values = [
                part.strip() for part in result.stdout.splitlines()[0].split(",")
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    driver = gpu_query_values[0] if len(gpu_query_values) == len(gpu_query_fields) else "N/A"
    lines.append(f"- **Driver version**: {driver}")

    if len(gpu_query_values) == len(gpu_query_fields):
        sm_clock, mem_clock, app_sm_clock, app_mem_clock = gpu_query_values[1:]
        lines.append(
            "- **GPU clocks**: "
            f"SM current {sm_clock} MHz, memory current {mem_clock} MHz, "
            f"application SM {app_sm_clock} MHz, "
            f"application memory {app_mem_clock} MHz"
        )

    return lines


class BenchmarkBase(Generic[W], ABC):
    """Abstract base class for op benchmarking.

    Generic over workload type so subclasses can declare the exact
    capability they need.  ``WorkloadBase`` remains the typical in-repo
    implementation, but the public contract is the type parameter.

    Subclass must implement calculate_flops() and calculate_memory().
    """

    def __init__(self, workload: W):
        self.workload = workload

    @abstractmethod
    def calculate_flops(self) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def calculate_memory(self) -> Optional[float]:
        raise NotImplementedError

    def profile(self,
                functor: Any,
                *inputs: Any) -> dict:
        """Profile a callable and return structured results.

        Uses the NVIDIA SOL-ExecBench protocol: CUPTI kernel timing,
        first-call plus warmup, 50 repeats × 3 trials, and SOL-style
        L2 preparation outside the timed window.
        """
        with torch.no_grad():
            latency = bench_kernel(functor, args=inputs)
        return self._build_result(latency)

    def profile_autograd(self, functor: Any) -> dict:
        """Profile a callable that requires autograd (e.g. fwd+bwd).

        Same as profile() but without torch.no_grad(), so the callable
        can build autograd graphs and call .backward() internally.
        The functor must be a zero-arg closure that captures its inputs.
        """
        latency = bench_kernel(functor)
        return self._build_result(latency)

    def _build_result(self, latency: float) -> dict:
        result = {"latency_ms": latency}
        # Deviations from the default protocol must be visible in reports.
        timing = getattr(_bench_meta, "timing", None)
        if timing is not None and timing != "cupti":
            result["timing"] = timing
        sampled_calls = getattr(_bench_meta, "cupti_sampled_calls", None)
        if sampled_calls is not None:
            result["cupti_sampled_calls"] = sampled_calls
        expected_kernel_count = getattr(_bench_meta, "cupti_expected_kernel_count", None)
        if expected_kernel_count is not None:
            result["cupti_expected_kernel_count"] = expected_kernel_count
        begin_tolerance = getattr(_bench_meta, "cupti_begin_tolerance_us", None)
        if begin_tolerance is not None:
            result["cupti_begin_tolerance_us"] = begin_tolerance
        end_tolerance = getattr(_bench_meta, "cupti_end_tolerance_us", None)
        if end_tolerance is not None:
            result["cupti_end_tolerance_us"] = end_tolerance
        repeat_guard = getattr(_bench_meta, "cupti_repeat_guard_us", None)
        if repeat_guard is not None:
            result["cupti_repeat_guard_us"] = repeat_guard
        input_policy = getattr(_bench_meta, "input_policy", None)
        if input_policy is not None:
            result["input_policy"] = input_policy
        input_policy_seed = getattr(_bench_meta, "input_policy_seed", None)
        if input_policy_seed is not None:
            result["input_policy_seed"] = input_policy_seed
        fallback_reason = getattr(_bench_meta, "fallback_reason", None)
        if fallback_reason is not None:
            result["fallback_reason"] = fallback_reason
        flops = self.calculate_flops()
        if flops is not None:
            result["tflops"] = flops / latency * 1e-9
        memory = self.calculate_memory()
        if memory is not None:
            result["bandwidth_tbs"] = memory / latency * 1e-9
        return result


# Manifest-driven benchmark helpers


def _workload_extra_params(w: dict, shape_key: str) -> dict[str, Any]:
    """Return op-call params on a workload entry, stripping reserved keys."""
    reserved = WORKLOAD_RESERVED_KEYS | {shape_key}
    return {
        k: v
        for k, v in w.items()
        if isinstance(k, str) and k not in reserved and not k.startswith("__")
    }


def workloads_to_params(op_name: str, include_extra: bool = False) -> list:
    """Convert manifest workload dicts for *op_name* to pytest params.

    Each entry becomes ``pytest.param(shape, dtype, id=...)``; with
    ``include_extra=True`` a third element carries the op-call params
    declared on the workload entry (e.g. ``{"dim": 0}``).
    """
    workloads = load_workloads(op_name)  # canonical not-found error
    shape_key, allowed = _workload_contract(op_name)
    params = []
    for w in workloads:
        if shape_key not in w:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} is missing "
                f"{shape_key!r} (derived from the signature's input name)."
            )
        unknown = sorted(
            repr(k) for k in w
            if not isinstance(k, str) or (k not in allowed and not k.startswith("__"))
        )
        if unknown:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} has unknown "
                f"keys {unknown}; allowed: {sorted(allowed)}."
            )
        shape = tuple(w[shape_key])
        label = w.get("label", "x".join(str(s) for s in shape))
        extra = _workload_extra_params(w, shape_key) if include_extra else {}
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            # Copy ``extra`` per parametrization so mutation in one test case
            # cannot leak into later cases sharing the workload entry.
            param_args = (
                (shape, dtype, dict(extra))
                if include_extra
                else (shape, dtype)
            )
            params.append(pytest.param(*param_args, id=f"{label}-{dtype_str}"))
    return params


def workload_field_params(workloads: list, keys: tuple) -> list:
    """Turn manifest workload dicts into pytest params.

    First workload is marked ``smoke``, the rest ``full``. Keys ending in
    ``dtype`` are resolved to ``torch.dtype`` values.
    """
    params = []
    for i, w in enumerate(workloads):
        args = [getattr(torch, w[k]) if k.endswith("dtype") else w[k] for k in keys]
        params.append(
            pytest.param(
                *args,
                marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                id=w["label"],
            )
        )
    return params


class ManifestBenchmark(BenchmarkBase[Any]):
    """Generic benchmark that reads FLOP/memory counts from an Op instance.

    Accepts an op name, an instantiated Op, and the workload that produced
    the inputs.  Roofline numbers come from ``op.eval_roofline()``, so the
    workload needs no ``shape`` / ``dtype`` metadata — it is retained only
    for subclasses that read their own fields off it.  Dynamic-shape ops may
    bind roofline variables during ``forward()``, so this helper calls
    ``op.eval_roofline()`` only while building a result after profiling has
    executed the op.

    Usage::

        op = SumFwdOp(dim=0)
        bm = ManifestBenchmark("SumFwdOp", op, workload)
        result = bm.profile(op, *inputs)
    """

    def __init__(
        self,
        op_name: str,
        op: Any,
        workload: Any,
    ):
        super().__init__(workload)
        self._op_name = op_name
        self._op = op
        self._roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _extract_op_config(op: object) -> Optional[dict]:
    """Return the kernel config for an Op instance, or None if unavailable.

    Handles the three Op patterns currently used in tileops:

      1. **Eager-init** (e.g. ``GemmOp``): ``op.kernel`` is a Kernel
         instance set in ``__init__``.
      2. **Lazy with dummy kernel** (e.g. ``FFTC2COp``): ``op.kernel`` is a
         default Kernel and ``op._kernel_cache`` may hold others.
      3. **Pure lazy cache** (e.g. ``_SoftmaxBaseOp`` and the spec-conformant
         reduction ops): ``op._kernel_cache`` is the only source; ``op.kernel``
         is unset.

    A direct ``op.config`` attribute (legacy / explicit override) takes
    precedence over kernel introspection.
    """
    op_config = getattr(op, "config", None)
    if op_config:
        return op_config

    kernel = getattr(op, "kernel", None)
    op_config = getattr(kernel, "config", None) if kernel is not None else None
    if op_config:
        return op_config

    # Pure lazy-cache pattern: pick any cached kernel's config. All cached
    # kernels for a given op share dtype/op_kind, so taking the first is
    # sufficient for the benchmark report (which records one entry per call).
    cache = getattr(op, "_kernel_cache", None)
    if cache:
        try:
            first_kernel = next(iter(cache.values()))
        except StopIteration:
            first_kernel = None
        if first_kernel is not None:
            op_config = getattr(first_kernel, "config", None)
            if op_config:
                return op_config

    return None


class BenchmarkReport:
    """Collects benchmark results and dumps a markdown report.

    All methods are static — use as BenchmarkReport.record(...).
    Call clear() at session start, dump() at session end.
    """
    _records: dict = {}

    @staticmethod
    def record(op_or_name, params: dict, result: dict, tag: str = "tileops") -> None:
        """Record a benchmark result.

        Args:
            op_or_name: Op instance or benchmark group name string.
                If an Op instance, class name and module are extracted automatically.
            params: Parameter dict (typically from locals())
            result: Dict with latency_ms, tflops, bandwidth_tbs
            tag: Label to distinguish implementations (e.g. "tileops", "FA3", "fla")
        """
        if isinstance(op_or_name, str):
            name = op_or_name
            op_module = None
            op_config = None
        else:
            name = op_or_name.__class__.__name__
            op_module = op_or_name.__class__.__module__
            op_config = _extract_op_config(op_or_name)

        # Filter params to only include serializable benchmark parameters.
        # Tuples of primitives (e.g. ``shape=(4096, 4096)``) are preserved
        # verbatim so the profile log carries the original input geometry
        # rather than a flattened element count.
        def _is_serializable(v: Any) -> bool:
            if isinstance(v, (int, float, bool, str, torch.dtype)):
                return True
            if isinstance(v, tuple):
                return all(_is_serializable(x) for x in v)
            return False

        filtered_params = {
            k: v for k, v in params.items()
            if k not in ("test", "bm", "op", "inputs", "result", "result_bl",
                         "baseline_fn", "tune")
            and not k.startswith("_")
            and _is_serializable(v)
        }
        record_entry = {
            "params": filtered_params,
            "result": result,
            "tag": tag,
        }
        if op_config:
            record_entry["config"] = op_config
        BenchmarkReport._records.setdefault(name, []).append(record_entry)

        # Accumulate in thread-local for conftest hook.
        if not hasattr(_bench_results, "entries"):
            _bench_results.entries = []
        entry = {"tag": tag, "op": name, **result}
        if op_module:
            entry["op_module"] = op_module
        _bench_results.entries.append(entry)

        _logger.info("op=%s module=%s tag=%s latency_ms=%.4f tflops=%.2f",
                      name, op_module or "N/A", tag,
                      result.get("latency_ms", 0),
                      result.get("tflops", 0))

    @staticmethod
    def dump(path: str) -> None:
        """Write all collected results to a markdown-formatted log file."""
        if not BenchmarkReport._records:
            return

        lines = [
            "# TileOPs Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Environment",
            "",
        ]
        lines.extend(_get_env_metadata())
        lines.append("")

        default_result_keys = ["latency_ms", "tflops", "bandwidth_tbs"]

        for name, entries in BenchmarkReport._records.items():
            if not entries:
                continue

            lines.append(f"## {name}")
            lines.append("")

            # Group by tag
            tag_entries = {}
            for entry in entries:
                tag_entries.setdefault(entry["tag"], []).append(entry)
            result_keys = list(default_result_keys)
            for entry in entries:
                for key in entry["result"]:
                    if key not in result_keys:
                        result_keys.append(key)

            for tag, tag_group in tag_entries.items():
                lines.append(f"### {tag}")
                lines.append("")

                param_keys = list(tag_group[0]["params"].keys())
                has_config = any("config" in e for e in tag_group)
                header_parts = param_keys + result_keys
                if has_config:
                    header_parts.append("config")
                lines.append("| " + " | ".join(header_parts) + " |")
                lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

                for entry in tag_group:
                    row = [str(entry["params"].get(k, "")) for k in param_keys]
                    for rk in result_keys:
                        val = entry["result"].get(rk)
                        if val is None:
                            row.append("N/A")
                        elif isinstance(val, (int, float)) and not isinstance(val, bool):
                            row.append(f"{val:.4f}")
                        else:
                            row.append(str(val))
                    if has_config:
                        cfg = entry.get("config")
                        row.append(str(cfg) if cfg else "")
                    lines.append("| " + " | ".join(row) + " |")

                lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        print(f"Benchmark report saved to {path}")

    @staticmethod
    def clear() -> None:
        """Clear all collected records."""
        BenchmarkReport._records.clear()
