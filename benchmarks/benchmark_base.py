import contextlib
import logging
import os
import random
import statistics
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
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
_cuda_runtime = None


class _NativeCUPTIAttributionError(Exception):
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


def _activity_identity(activity: dict) -> str:
    """Return a stable identity for a timed GPU activity."""
    kind = activity["kind"]
    if kind == "memcpy":
        return f"memcpy:{int(activity['copy_kind'])}:{int(activity['bytes'])}"
    if kind == "memset":
        return f"memset:{int(activity['bytes'])}:{int(activity['value'])}"
    return f"kernel:{activity['name']}"


def _kernel_sequence(kernels: list[dict]) -> tuple[str, ...]:
    return tuple(_activity_identity(activity) for activity in kernels)


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

    actual_sequence = _kernel_sequence(kernels)
    if actual_sequence == expected_sequence:
        return kernels
    if Counter(actual_sequence) != Counter(expected_sequence):
        return None

    # CUPTI may publish concurrently executing kernels in either start-time
    # order. Accept an inversion only when the two activities overlap; a
    # reordered serial launch is a real sequence change and must fail closed.
    expected_positions: dict[str, list[int]] = {}
    for position, name in enumerate(expected_sequence):
        expected_positions.setdefault(name, []).append(position)
    seen: Counter[str] = Counter()
    actual_to_expected = []
    for name in actual_sequence:
        occurrence = seen[name]
        actual_to_expected.append(expected_positions[name][occurrence])
        seen[name] += 1

    for left in range(expected_count):
        for right in range(left + 1, expected_count):
            if actual_to_expected[left] <= actual_to_expected[right]:
                continue
            left_kernel = kernels[left]
            right_kernel = kernels[right]
            if (
                int(left_kernel["end_ns"]) <= int(right_kernel["start_ns"])
                or int(right_kernel["end_ns"]) <= int(left_kernel["start_ns"])
            ):
                return None
    return kernels


def _kernel_span_us(kernels: list[dict]) -> float:
    if not kernels:
        return 0.0
    start_ns = min(int(kernel["start_ns"]) for kernel in kernels)
    end_ns = max(int(kernel["end_ns"]) for kernel in kernels)
    return (end_ns - start_ns) / 1000.0


def _format_sequence(seq: tuple[str, ...], limit: int = 8) -> str:
    """Render an activity sequence for an error message.

    An observed timing sequence spans every repeat, so cap the entry count as
    well as each mangled kernel name.
    """
    if not seq:
        return "<empty>"
    names = [name if len(name) <= 96 else name[:93] + "..." for name in seq[:limit]]
    if len(seq) > limit:
        names.append(f"...(+{len(seq) - limit} more)")
    return " -> ".join(names)


def _ordered_trace_kernels(trace: dict) -> list[dict]:
    return sorted(
        trace.get("kernels", []),
        key=lambda kernel: (int(kernel["start_ns"]), int(kernel["end_ns"])),
    )


def _stable_discovery_sequence(traces: list[dict], phase: str) -> tuple[str, ...]:
    for trace in traces:
        if int(trace.get("dropped", 0)) != 0:
            raise _NativeCUPTIAttributionError(
                f"CUPTI dropped {trace['dropped']} records during {phase} discovery"
            )
    groups = [_ordered_trace_kernels(trace) for trace in traces]
    sequences = [_kernel_sequence(kernels) for kernels in groups]
    if not sequences or any(not sequence for sequence in sequences):
        raise _NativeCUPTIAttributionError(
            f"CUPTI discovery found no CUDA kernel sequence for {phase}"
        )
    expected = sequences[0]
    if any(
        _select_expected_sequence(kernels, expected) is None
        for kernels in groups[1:]
    ):
        rendered = "; ".join(_format_sequence(sequence) for sequence in sequences)
        raise _NativeCUPTIAttributionError(
            f"CUPTI discovery saw inconsistent {phase} sequences: {rendered}"
        )
    return expected


def _discovery_repeats() -> int:
    return int(os.getenv("TILEOPS_CUPTI_DISCOVERY_REPEATS", "3"))


def _cuda_events_fallback_enabled() -> bool:
    return os.getenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0") == "1"


def _discover_expected_sequences(
    run_one: Callable[[int], None],
    prepare_one: Callable[[int], None],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    prepare_traces, operator_traces = native_cupti.collect_discovery(
        run_one,
        _discovery_repeats(),
        prepare_one,
    )
    return (
        _stable_discovery_sequence(prepare_traces, "prepare"),
        _stable_discovery_sequence(operator_traces, "operator"),
    )


def _attributed_latency_samples_ms(
    trace: dict,
    expected_sequence: tuple[str, ...],
    n_repeat: int,
    expected_prepare_sequence: tuple[str, ...] = (),
) -> list[float]:
    if int(trace.get("dropped", 0)) != 0:
        raise _NativeCUPTIAttributionError(
            f"CUPTI dropped {trace['dropped']} records during timing"
        )

    samples_us: list[float] = []
    kernels = _ordered_trace_kernels(trace)
    prepare_count = len(expected_prepare_sequence)
    operator_count = len(expected_sequence)
    cycle_count = prepare_count + operator_count
    expected_total = n_repeat * cycle_count

    if cycle_count == 0 or len(kernels) != expected_total:
        raise _NativeCUPTIAttributionError(
            "CUPTI timing activity count does not match the deterministic "
            f"op-sequence ledger: got {len(kernels)}, expected {expected_total} "
            f"({n_repeat} x ({prepare_count} prepare + {operator_count} operator)); "
            f"prepare=[{_format_sequence(expected_prepare_sequence)}]; "
            f"operator=[{_format_sequence(expected_sequence)}]; "
            f"observed=[{_format_sequence(_kernel_sequence(kernels))}]"
        )

    for repeat in range(n_repeat):
        begin = repeat * cycle_count
        prepare = kernels[begin:begin + prepare_count]
        operator = kernels[begin + prepare_count:begin + cycle_count]
        prepare_ok = (
            not expected_prepare_sequence
            or _select_expected_sequence(prepare, expected_prepare_sequence) is not None
        )
        selected = _select_expected_sequence(operator, expected_sequence)
        if not prepare_ok or selected is None:
            continue
        samples_us.append(_kernel_span_us(selected))

    if len(samples_us) != n_repeat:
        raise _NativeCUPTIAttributionError(
            f"CUPTI timing attributed {len(samples_us)}/{n_repeat} complete "
            f"expected kernel sequences; kernels={len(kernels)}; "
            f"prepare=[{_format_sequence(expected_prepare_sequence)}]; "
            f"operator=[{_format_sequence(expected_sequence)}]; "
            f"observed=[{_format_sequence(_kernel_sequence(kernels))}]"
        )
    return [sample_us * 1e-3 for sample_us in samples_us]


def _sample_spread_ms(samples: list[float]) -> tuple[float, float] | tuple[None, None]:
    """Return the 10th and 90th percentile of one op's timed samples.

    The reported latency is a median. Without the spread around it, a stable
    measurement and one dominated by launch jitter read the same downstream.
    """
    if len(samples) < 2:
        return None, None
    ordered = sorted(samples)
    last = len(ordered) - 1
    # Nearest rank, rounded rather than truncated: truncating collapses both
    # percentiles onto the minimum for small sample counts.
    return ordered[round(0.1 * last)], ordered[round(0.9 * last)]


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
) -> float:
    """Benchmark a GPU callable with CUPTI activity timing.

    Protocol (adapted from NVIDIA SOL-ExecBench, arxiv.org/abs/2603.19173):
      1. Lock GPU clocks externally (nvidia-smi).
      2. Run one first call plus *n_warmup* un-timed iterations, absorbing
         module/library lazy init, JIT/autotune, and allocator growth.
      3. Discover the per-call activity sequence, then profile one set of
         *n_repeat* iterations under CUPTI. Persisting L2 is reset and a
         2x-L2 buffer cleared before every iteration; inputs come from a
         shifting pool so each measured call sees a different data pointer
         without warming the flushed cache.
      4. Report the median per-iteration latency.

    Each timed call is measured from the earliest to the latest activity of
    its discovered sequence, so a single-activity call degenerates to pure
    kernel duration and a multi-activity call keeps its inter-kernel gaps.
    Attribution fails closed: a missing, unexpected, or dropped record raises
    unless ``TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1`` opts into diagnostic
    CUDA-event timing, which includes per-call launch overhead.

    Args:
        fn: Callable to benchmark.  If *args* is provided, called with
            tensors from the shifting input pool; otherwise called as ``fn()``.
        args: Tensor arguments to rotate each iteration.  Non-tensor
            values are passed through unchanged.
        n_warmup: Warmup iterations (default 10).
        n_repeat: Timed iterations (default 50).

    Returns:
        Kernel latency in **milliseconds**.
    """
    if not isinstance(args, tuple):
        raise TypeError(
            f"bench_kernel expects a tuple of args, got {type(args).__name__}. "
            "Check that gen_inputs() returns a tuple."
        )

    allow_fallback = _cuda_events_fallback_enabled()
    _bench_meta.timing = None
    _bench_meta.fallback_reason = None

    cache = _get_l2_flush_cache()

    if args:
        total_iterations = 1 + n_warmup + _discovery_repeats() + n_repeat
        if allow_fallback:
            total_iterations += n_repeat  # the fallback re-runs every repeat
        seed = int(os.getenv("TILEOPS_INPUT_POOL_SEED", "0"))
        arg_pool = _ShiftingTensorPool(args, total_iterations, seed=seed)
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

        def _prepare_args(i):
            return None

        def _run(i):
            return fn()

    def _prepare_iteration(i):
        _prepare_args(i)
        _reset_persisting_l2_cache()
        cache.zero_()
        torch.cuda.synchronize()

    # The first call and the warmup run outside every timing segment.
    _prepare_iteration(0)
    _run(0)
    torch.cuda.synchronize()

    for i in range(n_warmup):
        _prepare_iteration(i)
        _run(i)
    torch.cuda.synchronize()

    measurements_ms: list[float] = []
    try:
        with _native_output_suppressor():
            prepare_sequence, expected_sequence = _discover_expected_sequences(
                _run,
                _prepare_iteration,
            )
            trace = native_cupti.collect_repeats(
                _run,
                n_repeat,
                prepare_one=_prepare_iteration,
            )
            measurements_ms = _attributed_latency_samples_ms(
                trace,
                expected_sequence,
                n_repeat,
                prepare_sequence,
            )
        _bench_meta.timing = "native-cupti"
    except (_NativeCUPTIAttributionError, native_cupti.NativeCUPTIError) as exc:
        if not allow_fallback:
            raise RuntimeError(
                f"Native CUPTI profiling failed: {exc}. CUDA-events fallback is "
                "disabled (TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0), which keeps the "
                "run from silently mixing two timing methods."
            ) from exc
        _bench_meta.timing = "cuda-events"
        _bench_meta.fallback_reason = str(exc)
        _logger.warning(
            "Native CUPTI timing failed (%s); falling back to CUDA events, which "
            "include per-call launch and stream gaps.", exc,
        )
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        for i in range(n_repeat):
            _prepare_iteration(i)
            start_events[i].record()
            _run(i)
            end_events[i].record()
        torch.cuda.synchronize()
        measurements_ms = [
            start.elapsed_time(end)
            for start, end in zip(start_events, end_events, strict=True)
        ]

    latency_ms = statistics.median(measurements_ms)
    _bench_meta.latency_p10_ms, _bench_meta.latency_p90_ms = _sample_spread_ms(
        measurements_ms
    )
    # Drop the pool and its last handed-out view first: empty_cache() cannot
    # reclaim blocks these still reference, and the pools of hundreds of
    # benchmark calls would accumulate.
    arg_pool = None
    prepared_args = None
    torch.cuda.empty_cache()
    return latency_ms


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

        Uses the NVIDIA SOL-ExecBench protocol: CUPTI activity timing,
        first-call plus warmup, one set of 50 repeats, and SOL-style
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
        # How the number was measured, and how tightly, must travel with it: a
        # run that fell back to CUDA events is not comparable with a CUPTI-timed
        # one, and a median with a wide spread is not a latency.
        for key in ("timing", "fallback_reason", "latency_p10_ms", "latency_p90_ms"):
            value = getattr(_bench_meta, key, None)
            if value is not None:
                result[key] = value
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


def _op_kernels(op: object) -> Iterator[object]:
    """Yield the kernels *op* holds.

    An ``Op`` enumerates them itself through ``iter_kernels`` — slot entries and
    a directly bound ``op.kernel`` alike. A baseline that is not an ``Op``
    exposes at most a single ``kernel`` attribute.
    """
    iter_kernels = getattr(op, "iter_kernels", None)
    if callable(iter_kernels):
        yield from iter_kernels()
        return
    kernel = getattr(op, "kernel", None)
    if kernel is not None:
        yield kernel


def _extract_op_config(op: object) -> Optional[dict]:
    """Return the kernel config for an Op instance, or None if unavailable.

    A direct ``op.config`` attribute (explicit override) takes precedence over
    kernel introspection. Otherwise the first config among the kernels the op
    holds: kernels an op built share dtype and op kind, so the first is
    sufficient for a report that records one entry per call.
    """
    op_config = getattr(op, "config", None)
    if op_config:
        return op_config

    # FIXME(staged-rollout): reads `config` off the kernel object.
    #
    # Broken invariant: callers reach an op by calling it, not by reading its
    #   kernel's attributes.
    # Why: nothing reports what a call ran with; enumeration names the kernels
    #   but not what they were built for.
    # Cleanup: read the op's own execution metadata once it reports any.
    for kernel in _op_kernels(op):
        op_config = getattr(kernel, "config", None)
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
