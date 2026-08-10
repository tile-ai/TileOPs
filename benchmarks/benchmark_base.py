import contextlib
import logging
import os
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
)

import pytest
import torch
from torch.autograd.profiler import DeviceType

from tileops.manifest import (
    WORKLOAD_RESERVED_KEYS,
    load_manifest,
    load_workloads,
    single_input_workload_contract,
)


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


class _CuptiProjectionError(Exception):
    """CUPTI trace lacked a projected annotation window for every repeat."""


# Name of the ``record_function`` annotation wrapping the timed call. Kineto
# projects this scope onto the device timeline. The L2-flush ``cache.zero_()``
# is synchronized to completion before the window opens (see ``bench_kernel``),
# so its device event cannot fall inside a window regardless of how the
# projection behaves; kernels the timed call launches do.
_KERNEL_REGION = "tileops_bench_kernel"
_CUPTI_DUMMY_REGION = "tileops_bench_dummy"

# CUPTI 28 may return bad timestamps for a growing prefix of kernel activities
# after Kineto re-registers its timestamp callback for each profiler session.
# Sacrifice a dynamically sized prefix of unmeasured calls so the annotated
# measurement region remains complete. The learned count is retained on the
# benchmark thread across bench_kernel() calls because the faulty CUPTI state
# survives across profiler sessions in the same process.
_CUPTI_DUMMY_INITIAL = 1
_CUPTI_DUMMY_SAFETY_MARGIN = 2
_CUPTI_DUMMY_MAX = 100
_CUPTI_PROJECTION_RETRIES = 3
_CUPTI_EVENT_BUDGET = 10_000
_CUPTI_MIN_REPEATS = 5


def _effective_repeats(requested: int, kernel_count: int, n_trials: int) -> int:
    """Bound trace size without reducing ordinary one/few-kernel sampling."""
    if requested <= 0 or kernel_count <= 0 or n_trials <= 0:
        return requested
    budgeted = _CUPTI_EVENT_BUDGET // (kernel_count * n_trials)
    return min(requested, max(_CUPTI_MIN_REPEATS, budgeted))


def _call_signature(fn, args: tuple[Any, ...]) -> tuple:
    """Stable-enough in-process key for reusing activity fan-out counts."""
    owner = fn.__self__ if hasattr(fn, "__self__") else fn
    callable_id = (
        type(owner).__module__,
        type(owner).__qualname__,
        getattr(fn, "__qualname__", type(fn).__qualname__),
    )
    arg_signature = tuple(
        (tuple(arg.shape), str(arg.dtype), tuple(arg.stride()))
        if isinstance(arg, torch.Tensor) else (type(arg).__qualname__, repr(arg))
        for arg in args
    )
    return callable_id + (arg_signature,)


def _trial_means_ms(
    by_repeat_us: dict[int, float], repeats: int, trials: int
) -> list[float]:
    """Return one mean latency in milliseconds for each trial."""
    return [
        sum(by_repeat_us[i] for i in range(t * repeats, (t + 1) * repeats))
        / repeats
        * 1e-3
        for t in range(trials)
    ]


def _kernel_time_us_by_repeat(kineto_results):
    """Collect summed and elapsed device time for each timed call.

    Sums only kernels inside a :data:`_KERNEL_REGION` annotation window, so the
    L2-flush fill is excluded and the kernel under test is counted regardless of
    its name. A call launching several kernels contributes all of them.

    Iterates the C++ Kineto events directly to bypass ``key_averages()``, which
    is ~16x slower (~130ms of Python parsing/tree-building) for large traces.

    Returns:
        ``(sum_by_repeat, span_by_repeat, counts, repeat_indices)``. ``sum`` is
        the sum of all attributed kernel durations, while ``span`` is the time
        from the first attributed kernel start to the last attributed kernel
        end. Both are in microseconds and come from the same Kineto trace.
    """
    import bisect

    windows: list[tuple[int, int, Optional[int]]] = []
    kernels: list[tuple[int, int]] = []  # (start_ns, duration_ns)
    for evt in kineto_results.events():
        if evt.device_type() != DeviceType.CUDA:
            continue
        if evt.is_user_annotation():
            name = evt.name()
            if name == _KERNEL_REGION or name.startswith(f"{_KERNEL_REGION}:"):
                repeat_index = None
                if name != _KERNEL_REGION:
                    with contextlib.suppress(ValueError):
                        repeat_index = int(name.rsplit(":", 1)[1])
                windows.append((evt.start_ns(), evt.end_ns(), repeat_index))
            continue
        # Kineto exposes memcpy/memset records as CUDA device activities too.
        # Native CUPTI's kernel collector excludes them, so do not label or
        # sum those records as kernels here.
        activity_name = evt.name()
        if activity_name.startswith(("Memcpy", "Memset")):
            continue
        kernels.append((evt.start_ns(), evt.duration_ns()))

    windows.sort()
    starts = [w[0] for w in windows]
    ends = [w[1] for w in windows]
    by_repeat: dict[int, float] = {}
    bounds: dict[int, tuple[int, int]] = {}
    counts: dict[int, int] = {}
    for start_ns, dur_ns in kernels:
        # Count only kernels that fall inside a timed-call window; everything
        # outside (notably the L2-flush fill) is excluded.
        idx = bisect.bisect_right(starts, start_ns) - 1
        if idx >= 0 and start_ns < ends[idx] and windows[idx][2] is not None:
            repeat_index = windows[idx][2]
            assert repeat_index is not None
            by_repeat[repeat_index] = by_repeat.get(repeat_index, 0.0) + dur_ns / 1000.0
            end_ns = start_ns + dur_ns
            if repeat_index in bounds:
                first_ns, last_ns = bounds[repeat_index]
                bounds[repeat_index] = (min(first_ns, start_ns), max(last_ns, end_ns))
            else:
                bounds[repeat_index] = (start_ns, end_ns)
            counts[repeat_index] = counts.get(repeat_index, 0) + 1
    spans = {
        repeat_index: (last_ns - first_ns) / 1000.0
        for repeat_index, (first_ns, last_ns) in bounds.items()
    }
    return by_repeat, spans, counts, [w[2] for w in windows]


def _projected_region_indices(kineto_results, region: str) -> list[int]:
    """Return indices of projected GPU annotations for *region*."""
    prefix = f"{region}:"
    projected = []
    for evt in kineto_results.events():
        if evt.device_type() != DeviceType.CUDA or not evt.is_user_annotation():
            continue
        name = evt.name()
        if not name.startswith(prefix):
            continue
        with contextlib.suppress(ValueError):
            projected.append(int(name.rsplit(":", 1)[1]))
    return projected


def _next_dummy_count(
    dummy_count: int,
    projected_dummy_indices: list[int],
    projected_repeat_indices: list[int],
    n_repeat: int,
) -> int:
    """Size the next sacrificial prefix from this profiler's missing regions."""
    projected_dummy = len(set(projected_dummy_indices))
    projected_repeat = len(set(projected_repeat_indices))
    missing_prefix = max(0, dummy_count - projected_dummy) + max(0, n_repeat - projected_repeat)
    return min(
        _CUPTI_DUMMY_MAX,
        max(_CUPTI_DUMMY_INITIAL, missing_prefix + _CUPTI_DUMMY_SAFETY_MARGIN),
    )


# L2 cache flush buffer (sized to actual L2, allocated lazily)

_l2_flush_cache: Optional[torch.Tensor] = None


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
        _l2_flush_cache = torch.empty(l2_bytes // 4, dtype=torch.int, device="cuda")
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
    """Benchmark a GPU kernel with pure kernel timing via CUPTI.

    Protocol (adapted from NVIDIA SOL-ExecBench, arxiv.org/abs/2603.19173):
      1. Lock GPU clocks externally (nvidia-smi).
      2. Run *n_warmup* un-timed iterations with L2 flush.
      3. For each of *n_trials* trials, profile *n_repeat* iterations
         under CUPTI to get pure kernel execution time (no launch overhead).
         L2 is flushed before every iteration.  Input tensors are cloned
         each iteration so the kernel always sees fresh addresses.
      4. Report the median trial mean (robust to outlier trials).

    Uses CUPTI via torch.profiler for accurate kernel-only timing, with
    direct Kineto C++ event iteration to avoid Python parsing overhead.
    Falls back to CUDA events if CUPTI is unavailable.

    Args:
        fn: Callable to benchmark.  If *args* is provided, called as
            ``fn(*cloned_args)``; otherwise called as ``fn()``.
        args: Tensor arguments to clone each iteration.  Non-tensor
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

    # Thread-local state survives across benchmark calls in the same pytest
    # process, so clear the previous case's fallback diagnostic up front.
    _bench_meta.fallback_error = None
    _bench_meta.cupti_dummy_calls = []
    _bench_meta.cupti_corrupted_prefixes = []
    _bench_meta.cupti_projection_retries = 0
    _bench_meta.cupti_dummy_history = []
    _bench_meta.cupti_kernel_count = None
    _bench_meta.cupti_effective_repeats = n_repeat
    _bench_meta.cupti_discovery_ms = 0.0
    _bench_meta.cupti_collect_ms = 0.0
    _bench_meta.kernel_sum_ms = None
    _bench_meta.kernel_span_ms = None
    if not hasattr(_bench_meta, "cupti_kernel_count_cache"):
        _bench_meta.cupti_kernel_count_cache = {}
    if not hasattr(_bench_meta, "cupti_dummy_count"):
        _bench_meta.cupti_dummy_count = _CUPTI_DUMMY_INITIAL

    cache = _get_l2_flush_cache()
    has_args = len(args) > 0

    # Pre-clone a small pool of input tensors so the kernel sees different
    # addresses across iterations.  Skip cloning if total tensor memory
    # exceeds 1 GB to avoid OOM on large workloads.
    _N_CLONES = 3
    _MAX_CLONE_BYTES = 1 << 30  # 1 GB
    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(a.nelement() * a.element_size()
                          for a, m in zip(args, tensor_mask, strict=True) if m)
        if total_bytes * _N_CLONES <= _MAX_CLONE_BYTES:
            arg_pool = [
                tuple(a.clone() if m else a for a, m in zip(args, tensor_mask, strict=True))
                for _ in range(_N_CLONES)
            ]
            def _run(i):
                return fn(*arg_pool[i % _N_CLONES])
        else:
            _logger.warning(
                "bench_kernel: inputs total %.2f GiB; skipping per-iteration "
                "cloning (kernel sees identical addresses)",
                total_bytes / (1 << 30),
            )
            arg_pool = None
            def _run(i):
                return fn(*args)
    else:
        arg_pool = None
        def _run(i):
            return fn()
    _bench_meta.inputs_cloned = arg_pool is not None or not has_args

    # Warmup (no profiling)
    for i in range(n_warmup):
        cache.zero_()
        _run(i)
    torch.cuda.synchronize()

    # torch.profiler.schedule is avoided because queued launches can leak
    # across its warmup/active boundary. Drain the L2 flush before opening each
    # projected call window and drain the call before the next flush, so every
    # attributed device activity belongs to exactly one repeat.
    trial_means: list[float] = []
    trial_span_means: list[float] = []
    try:
        with _native_output_suppressor():
            signature = _call_signature(fn, args)
            kernel_count = _bench_meta.cupti_kernel_count_cache.get(signature)
            probe_means = None
            probe_span_means = None
            discovery_start = time.monotonic()
            for _discovery_attempt in range(
                _CUPTI_PROJECTION_RETRIES + 1 if kernel_count is None else 0
            ):
                dummy_count = _bench_meta.cupti_dummy_count
                probe_repeats = _CUPTI_MIN_REPEATS * n_trials
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                ) as discovery:
                    for i in range(dummy_count):
                        cache.zero_()
                        torch.cuda.synchronize()
                        with torch.profiler.record_function(f"{_CUPTI_DUMMY_REGION}:{i}"):
                            _run(i)
                        torch.cuda.synchronize()
                    for repeat_index in range(probe_repeats):
                        cache.zero_()
                        torch.cuda.synchronize()
                        with torch.profiler.record_function(
                            f"{_KERNEL_REGION}:{repeat_index}"
                        ):
                            _run(repeat_index)
                        torch.cuda.synchronize()
                (
                    discovered_times,
                    discovered_spans,
                    discovered_counts,
                    discovered_indices,
                ) = _kernel_time_us_by_repeat(discovery.profiler.kineto_results)
                counts = [discovered_counts.get(i, 0) for i in range(probe_repeats)]
                projected_repeats = {
                    i for i in discovered_indices if i is not None
                }
                if (
                    projected_repeats == set(range(probe_repeats))
                    and counts[0] > 0
                    and len(set(counts)) == 1
                ):
                    kernel_count = counts[0]
                    _bench_meta.cupti_kernel_count_cache[signature] = kernel_count
                    if _effective_repeats(n_repeat, kernel_count, n_trials) == _CUPTI_MIN_REPEATS:
                        probe_means = _trial_means_ms(
                            discovered_times, _CUPTI_MIN_REPEATS, n_trials
                        )
                        probe_span_means = _trial_means_ms(
                            discovered_spans, _CUPTI_MIN_REPEATS, n_trials
                        )
                    break
                _bench_meta.cupti_dummy_count = min(
                    _CUPTI_DUMMY_MAX, dummy_count + _CUPTI_DUMMY_SAFETY_MARGIN
                )
            if kernel_count is None:
                raise _CuptiProjectionError("discovery call was not projected")
            _bench_meta.cupti_discovery_ms = (time.monotonic() - discovery_start) * 1e3
            effective_repeats = _effective_repeats(n_repeat, kernel_count, n_trials)
            _bench_meta.cupti_kernel_count = kernel_count
            _bench_meta.cupti_effective_repeats = effective_repeats
            total_repeats = n_trials * effective_repeats
            if probe_means is not None:
                trial_means = probe_means
                assert probe_span_means is not None
                trial_span_means = probe_span_means
                _bench_meta.cupti_dummy_calls.append(dummy_count)
                _bench_meta.cupti_dummy_history.append(
                    f"probe-reused:dummy={dummy_count},timed={total_repeats}/{total_repeats},ok=1"
                )
            collect_start = time.monotonic()
            for attempt in range(_CUPTI_PROJECTION_RETRIES + 1 if not trial_means else 0):
                dummy_count = _bench_meta.cupti_dummy_count
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                ) as profiler:
                    # CUPTI's corrupt prefix occurs when a profiler session is
                    # registered. Pay that cost once, then collect every trial
                    # in the same session with globally unique repeat IDs.
                    for i in range(dummy_count):
                        cache.zero_()
                        torch.cuda.synchronize()
                        with torch.profiler.record_function(f"{_CUPTI_DUMMY_REGION}:{i}"):
                            _run(i)
                        torch.cuda.synchronize()
                    for repeat_index in range(total_repeats):
                        cache.zero_()
                        torch.cuda.synchronize()
                        with torch.profiler.record_function(
                            f"{_KERNEL_REGION}:{repeat_index}"
                        ):
                            _run(repeat_index)
                        torch.cuda.synchronize()

                kineto_results = profiler.profiler.kineto_results
                by_repeat, spans_by_repeat, _, repeat_indices = (
                    _kernel_time_us_by_repeat(kineto_results)
                )
                projected = [i for i in repeat_indices if i is not None]
                dummy_indices = _projected_region_indices(kineto_results, _CUPTI_DUMMY_REGION)
                _bench_meta.cupti_dummy_count = _next_dummy_count(
                    dummy_count, dummy_indices, projected, total_repeats
                )

                missing = sorted(set(range(total_repeats)) - set(projected))
                missing_dummy = max(0, dummy_count - len(set(dummy_indices)))
                corrupted_prefix = missing_dummy + len(missing)
                _bench_meta.cupti_corrupted_prefixes.append(corrupted_prefix)
                projection_complete = (
                    not missing
                    and len(by_repeat) == total_repeats
                )
                _bench_meta.cupti_dummy_history.append(
                    f"all-trialsa{attempt + 1}:dummy={dummy_count},"
                    f"corrupt={corrupted_prefix},"
                    f"timed={len(set(projected))}/{total_repeats},"
                    f"ok={int(projection_complete)}"
                )
                if projection_complete:
                    _bench_meta.cupti_dummy_calls.append(dummy_count)
                    trial_means = _trial_means_ms(
                        by_repeat, effective_repeats, n_trials
                    )
                    trial_span_means = _trial_means_ms(
                        spans_by_repeat, effective_repeats, n_trials
                    )
                    break

                n_cuda_kernels = sum(
                    1
                    for evt in kineto_results.events()
                    if evt.device_type() == DeviceType.CUDA and not evt.is_user_annotation()
                )
                _logger.debug(
                    "CUPTI projection mismatch on attempt %d/%d: %d/%d annotation "
                    "windows, %d dummies, next=%d (%d CUDA kernels captured)",
                    attempt + 1,
                    _CUPTI_PROJECTION_RETRIES + 1,
                    len(projected),
                    total_repeats,
                    dummy_count,
                    _bench_meta.cupti_dummy_count,
                    n_cuda_kernels,
                )
                _bench_meta.cupti_projection_retries += 1
                if attempt == _CUPTI_PROJECTION_RETRIES:
                    details = [
                        f"attempt {attempt + 1}/{_CUPTI_PROJECTION_RETRIES + 1}",
                        f"{len(projected)}/{total_repeats} annotation windows projected",
                        f"dummy calls={dummy_count}",
                        f"next dummy calls={_bench_meta.cupti_dummy_count}",
                        f"missing repeats (1-based)={[i + 1 for i in missing]}",
                    ]
                    details.append(f"{n_cuda_kernels} CUDA kernels captured")
                    raise _CuptiProjectionError(", ".join(details))
            _bench_meta.cupti_collect_ms = (time.monotonic() - collect_start) * 1e3
        _bench_meta.timing = "cupti"
    except _CuptiProjectionError as exc:
        _bench_meta.fallback_error = str(exc)
        # Check if cuda-events fallback is allowed
        allow_fallback = os.getenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "1") == "1"

        if not allow_fallback:
            raise RuntimeError(
                f"CUPTI profiling failed: {exc}. "
                "CUDA-events fallback is disabled (TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0). "
                "This prevents generating inaccurate benchmark data with ~7x inflated latency. "
                "To debug: run with TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1 and check logs."
            ) from exc

        _logger.warning(
            "CUPTI projection failed (%s); falling back to CUDA-events "
            "timing, which includes ~50-60us launch overhead per call. "
            "Latency will be inflated by ~6-7x for fast kernels (<10us). "
            "Set TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0 to prevent fallback.",
            exc,
        )
        trial_means = []
        trial_span_means = []

    # Fallback to CUDA events if CUPTI failed
    if not trial_means:
        _bench_meta.timing = "cuda-events"
        # Mimic CUPTI behavior: flush L2 before measurement window
        for _ in range(n_trials):
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

            for i in range(n_repeat):
                cache.zero_()
                torch.cuda.synchronize()  # Drain flush before measurement
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
    latency = trial_means[len(trial_means) // 2]
    if _bench_meta.timing == "cupti":
        trial_span_means.sort()
        _bench_meta.kernel_sum_ms = latency
        _bench_meta.kernel_span_ms = trial_span_means[len(trial_span_means) // 2]
    return latency


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
        10 warmup, 50 repeats × 3 trials, L2 flush sized to actual
        cache, input tensors cloned each iteration.
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
        kernel_sum_ms = getattr(_bench_meta, "kernel_sum_ms", None)
        kernel_span_ms = getattr(_bench_meta, "kernel_span_ms", None)
        if kernel_sum_ms is not None and kernel_span_ms is not None:
            result["kernel_sum_ms"] = kernel_sum_ms
            result["kernel_span_ms"] = kernel_span_ms
            # Keep latency_ms backward compatible while making its meaning
            # explicit to report consumers.
            result["latency_metric"] = "kernel_sum"
        # Deviations from the default protocol must be visible in reports.
        timing = getattr(_bench_meta, "timing", None)
        if timing is not None and timing != "cupti":
            result["timing"] = timing
        fallback_error = getattr(_bench_meta, "fallback_error", None)
        if fallback_error is not None:
            result["fallback_error"] = fallback_error
        dummy_calls = getattr(_bench_meta, "cupti_dummy_calls", [])
        if dummy_calls:
            result["cupti_dummy_calls"] = max(dummy_calls)
        corrupted_prefixes = getattr(_bench_meta, "cupti_corrupted_prefixes", [])
        if corrupted_prefixes:
            result["cupti_corrupted_prefix"] = max(corrupted_prefixes)
        projection_retries = getattr(_bench_meta, "cupti_projection_retries", 0)
        result["cupti_projection_retries"] = projection_retries
        kernel_count = getattr(_bench_meta, "cupti_kernel_count", None)
        if kernel_count is not None:
            result["cupti_kernel_count"] = kernel_count
        result["cupti_effective_repeats"] = getattr(
            _bench_meta, "cupti_effective_repeats", 0
        )
        result["cupti_discovery_ms"] = getattr(_bench_meta, "cupti_discovery_ms", 0.0)
        result["cupti_collect_ms"] = getattr(_bench_meta, "cupti_collect_ms", 0.0)
        dummy_history = getattr(_bench_meta, "cupti_dummy_history", [])
        if dummy_history:
            result["cupti_dummy_history"] = ";".join(dummy_history)
        if getattr(_bench_meta, "inputs_cloned", True) is False:
            result["inputs_cloned"] = False
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
