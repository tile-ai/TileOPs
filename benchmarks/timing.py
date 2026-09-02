"""How a call is timed: CUPTI activity collection, and the loop around it.

The timing layer owns the measurement and nothing else -- it knows how to run a callable
n times and return each run's device latency. What the numbers mean, and where they are
written, belong to the layers above.

A kernel belongs to the iteration whose external correlation id it carries, so nothing
is inferred from timestamps and a kernel shorter than the host overhead around it is
attributed as reliably as a long one. CUPTI's id stack is per-thread, which is this
protocol's one demand on a timed callable: it must launch its own work rather than hand
it to another thread (``Tensor.backward`` hands it to autograd's engine thread;
``grad_fn.apply`` does not).
"""

import contextlib
import ctypes
import logging
import os
import statistics
import sys
import threading
from typing import Any, Callable, NamedTuple, Optional

import torch

_logger = logging.getLogger("tileops.bench")

# Per-phase wall-time budgets in ms, divided by the cost of one cold-isolated iteration.
# That cost includes the L2 flush, which the reading excludes and which on its own
# outweighs a microsecond kernel, so most ops take their count from _MAX_ITERS instead.
DRY_RUN_MS = 25.0
REPEAT_MS = 100.0
_CALIBRATION_ITERS = 3
# Counts are clamped: attribution requires every repeat to reach the GPU, so an
# unbounded count turns one hiccup into a failed case.
_MIN_ITERS = 10
_MAX_ITERS = 200

# Latest bench_kernel measurement metadata; deviations from the default protocol are
# surfaced in results by BenchmarkBase._build_result.
_bench_meta = threading.local()
_cuda_runtime = None

# CUPTI activity collection, via NVIDIA's cupti-python binding.

_CUPTI = None
_COLLECTOR_ACTIVE = False
_CALLBACKS_REGISTERED = False
# Whatever CUPTI does with a buffer between handing it back and asking for the next one
# scales with its size and runs on this thread, inside a timed call, so a buffer of a
# few megabytes stalls iterations of a short call. Only latency_ms picks the stall up;
# the records are the same either way.
_BUFFER_BYTES = 256 * 1024
_BUFFER_ALIGN = 8
# What the next buffer request answers with. A phase that lost records raises it for
# the retry: a stall costs one iteration's latency_ms, a lost record costs the case.
_buffer_bytes = _BUFFER_BYTES
_KERNELS: list[dict[str, Any]] = []
_ITERATION_OF: dict[int, int] = {}
_ATTRIBUTION_ATTEMPTS = 3
_DROPPED_COUNT_UNREADABLE = False
_DROP_COUNTER_LIVE: Optional[bool] = None
# Pushed around the L2 flush and anything else run between iterations. Labelling that
# work is what lets a kernel with no id at all mean one thing only: a thread the id was
# never pushed on launched it. Out of range of any iteration index.
_PREPARE_ID = 1 << 32


# L2 cache flush buffer, allocated lazily.
_l2_flush_cache: Optional[torch.Tensor] = None


def _clamp_iters(raw: float, max_iters: int = _MAX_ITERS, min_iters: int = _MIN_ITERS) -> int:
    return max(min_iters, min(max_iters, int(raw)))


class _CUPTIAttributionError(Exception):
    """A CUPTI trace could not be attributed to logical benchmark calls."""


class _CUPTIRecordsLostError(_CUPTIAttributionError):
    """A reading was taken and then discarded. Its own class because it can be retried."""


class _OffThreadLaunchError(_CUPTIAttributionError):
    """Kernels ran that no iteration claims, so a thread without the id launched them."""


class CUPTIError(RuntimeError):
    """The CUPTI collector is unavailable or could not be operated."""


def _load_cupti():
    global _CUPTI
    if _CUPTI is not None:
        return _CUPTI
    try:
        from cupti import cupti
    except Exception as exc:  # noqa: BLE001
        raise CUPTIError(
            "cupti-python is unavailable. Install it with "
            "`pip install --no-deps cupti-python==13.2.0`; --no-deps is required "
            "or it downgrades torch's cuda-bindings pin."
        ) from exc
    _CUPTI = cupti
    return _CUPTI


def _buffer_requested():
    return _buffer_bytes, _BUFFER_ALIGN


def _read_dropped() -> Optional[int]:
    """Read CUPTI's drop counter, which resets on every read. None if the call fails."""
    global _DROPPED_COUNT_UNREADABLE
    cupti = _load_cupti()
    dropped = ctypes.c_size_t(0)
    try:
        # Kernel records land on the global queue, which is context 0, and the binding
        # takes the out-parameter as an address rather than returning it.
        cupti.activity_get_num_dropped_records(0, 0, ctypes.addressof(dropped))
    except Exception as exc:  # noqa: BLE001
        if not _DROPPED_COUNT_UNREADABLE:
            _DROPPED_COUNT_UNREADABLE = True
            _logger.warning(
                "CUPTI dropped-record count is unreadable (%s); an unmeasured iteration "
                "can no longer be told apart from a lost record.",
                exc,
            )
        return None
    return int(dropped.value)


def _drop_counter_is_live() -> bool:
    """Prove the drop counter reports a loss, by causing one, before trusting it.

    A counter stuck at zero -- wrong queue, unimplemented in this binding -- would turn
    every lost record into "the call never reached the device", the misdiagnosis this
    path exists to prevent. Measured once per process rather than assumed.
    """
    global _DROP_COUNTER_LIVE
    if _DROP_COUNTER_LIVE is not None:
        return _DROP_COUNTER_LIVE
    probe_kernels = 4
    probe = torch.empty(1, device="cuda")
    with _phase_session(buffer_bytes=8):  # too small to hold one record
        for _ in range(probe_kernels):
            probe.zero_()
        torch.cuda.synchronize()
        _load_cupti().activity_flush_all(1)
    dropped = _read_dropped()
    _DROP_COUNTER_LIVE = bool(dropped)
    if not _DROP_COUNTER_LIVE:
        _logger.warning(
            "CUPTI reported %s dropped records after discarding %d on purpose; the "
            "counter cannot be trusted, so an unmeasured iteration will not be blamed "
            "on the call.",
            dropped,
            probe_kernels,
        )
    return _DROP_COUNTER_LIVE


def _buffer_completed(records) -> None:
    # Copy the fields out and keep no record alive: the binding's other
    # accessors misread a newer libcupti's struct and raise, including from
    # __del__ at shutdown.
    cupti = _load_cupti()
    for record in records:
        kind = int(record.kind)
        if kind == int(cupti.ActivityKind.CONCURRENT_KERNEL):
            _KERNELS.append(
                {
                    "name": str(record.name),
                    "start_ns": int(record.start),
                    "end_ns": int(record.end),
                    "correlation_id": int(record.correlation_id),
                }
            )
        elif kind == int(cupti.ActivityKind.MEMCPY):
            _KERNELS.append(
                {
                    "name": "memcpy",
                    "start_ns": int(record.start),
                    "end_ns": int(record.end),
                    "correlation_id": int(record.correlation_id),
                    "is_copy": True,
                }
            )
        elif kind == int(cupti.ActivityKind.EXTERNAL_CORRELATION):
            _ITERATION_OF[int(record.correlation_id)] = int(record.external_id)
        # Launch-API records are collected only so CUPTI emits the correlation above.


def _trace_launches_only(cupti) -> None:
    """Silence every traced API except the ones that issue the work being timed.

    Tracing all of them costs ~15 records per iteration instead of one, and only a launch
    carries the correlation id a kernel needs. Graph launches count: a replayed graph's
    kernels are unattributable without ``cudaGraphLaunch`` / ``cuGraphLaunch``. Must run
    after ``activity_enable``, which resets the per-API filter -- measured, not assumed.

    The memcpy APIs are traced whether or not the case counts copies: a copy activity
    carries the correlation id of the call that issued it, and without that id a copy can
    neither be counted nor reported as uncounted.
    """
    issuers = ("Launch", "Memcpy")
    for toggle, cbids in (
        (cupti.activity_enable_runtime_api, cupti.runtime_api_trace_cbid),
        (cupti.activity_enable_driver_api, cupti.driver_api_trace_cbid),
    ):
        for name in dir(cbids):
            cbid = getattr(cbids, name, None)
            if name.startswith("_") or not isinstance(cbid, int):
                continue
            try:
                toggle(int(cbid), 1 if any(word in name for word in issuers) else 0)
            except Exception:  # noqa: BLE001, S112
                # The enums carry sentinels (INVALID, SIZE) that CUPTI refuses.
                continue


@contextlib.contextmanager
def _phase_session(buffer_bytes: int = _BUFFER_BYTES):
    """Own one session, so a failed trial leaves nothing behind for the next."""
    global _COLLECTOR_ACTIVE, _CALLBACKS_REGISTERED, _buffer_bytes
    if _COLLECTOR_ACTIVE:
        raise RuntimeError("CUPTI collector is already active")
    cupti = _load_cupti()
    # Copies are always collected and counted only where the case asks, since only the
    # case knows whether a copy is the arithmetic.
    kinds = [
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.EXTERNAL_CORRELATION,
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.MEMCPY,
    ]
    previous_bytes = _buffer_bytes
    try:
        if not _CALLBACKS_REGISTERED:
            cupti.activity_register_callbacks(_buffer_requested, _buffer_completed)
            _CALLBACKS_REGISTERED = True
        _buffer_bytes = buffer_bytes
        _KERNELS.clear()
        _ITERATION_OF.clear()
        # Read and discard, so the count taken after the flush belongs to this phase
        # alone. Raw, because proving the counter live opens a session of its own.
        _read_dropped()
        for kind in kinds:
            cupti.activity_enable(kind)
        _trace_launches_only(cupti)
    except Exception as exc:  # noqa: BLE001
        _buffer_bytes = previous_bytes
        raise CUPTIError(f"CUPTI collector failed to start: {exc}") from exc
    _COLLECTOR_ACTIVE = True
    try:
        yield
    finally:
        _COLLECTOR_ACTIVE = False
        _buffer_bytes = previous_bytes
        try:
            for kind in reversed(kinds):
                cupti.activity_disable(kind)
        except Exception as exc:  # noqa: BLE001
            raise CUPTIError(f"CUPTI collector failed to stop: {exc}") from exc


def _flush() -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Return the kernels recorded since the previous flush, and their iteration map."""
    cupti = _load_cupti()
    torch.cuda.synchronize()
    try:
        cupti.activity_flush_all(1)  # CUPTI_ACTIVITY_FLAG_FLUSH_FORCED
    except Exception as exc:  # noqa: BLE001
        raise CUPTIError(f"CUPTI flush failed: {exc}") from exc
    kernels, iteration_of = list(_KERNELS), dict(_ITERATION_OF)
    _KERNELS.clear()
    _ITERATION_OF.clear()
    return kernels, iteration_of


class Trace(NamedTuple):
    """One phase of collection: what CUPTI recorded, and what it lost."""

    kernels: list[dict[str, Any]]
    iteration_of: dict[int, int]
    """Which iteration each correlation id belongs to."""
    dropped: Optional[int]
    """Records discarded for want of buffer space, or None if the count is unproven."""


def collect_repeats(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None],
    buffer_bytes: int = _BUFFER_BYTES,
) -> Trace:
    """Run the timed repeats, each labelled with its iteration index.

    ``prepare_one`` runs under an id of its own and leaves the device idle, and each
    call is drained before the next begins, so an iteration's kernels neither overlap
    nor borrow a neighbour's cache state.
    """
    cupti = _load_cupti()
    kind = cupti.ExternalCorrelationKind.CUSTOM0
    # Before the session, not inside it: the proof runs a phase of its own.
    _drop_counter_is_live()

    @contextlib.contextmanager
    def _labelled(external_id: int):
        cupti.activity_push_external_correlation_id(kind, external_id)
        try:
            yield
        finally:
            cupti.activity_pop_external_correlation_id(kind)

    with _phase_session(buffer_bytes):
        for i in range(n_repeat):
            with _labelled(_PREPARE_ID):
                prepare_one(i)
            with _labelled(i):
                run_one(i)
            torch.cuda.synchronize()
        kernels, iteration_of = _flush()
        # Counted after the flush, once CUPTI has handed back what it did keep. Unproven
        # counter reports None, not zero: it is the only evidence separating a lost
        # reading from a call that launched nothing.
        dropped = _read_dropped() if _drop_counter_is_live() else None
        return Trace(kernels, iteration_of, dropped)


class Sample(NamedTuple):
    """One iteration's reading."""

    device_busy_ms: float
    """Union of the call's kernel execution intervals. Host-independent."""
    latency_ms: float
    """Earliest kernel start to latest kernel end. Includes gaps the host caused."""
    n_kernels: int | None
    """Kernels attributed to the call, or None when the timer cannot see them.

    A copy counts as one where the case asked for copies; see ``count_copies``.
    """
    uncounted_copy_ms: float = 0.0
    """Device time in copies the call issued and this case left out of the reading.

    Zero for a case that counts copies, and for one that issues none. Anything else is
    a reading that omits work: see ``count_copies``.
    """


def _kernel_span_us(kernels: list[dict]) -> float:
    if not kernels:
        return 0.0
    start_ns = min(int(kernel["start_ns"]) for kernel in kernels)
    end_ns = max(int(kernel["end_ns"]) for kernel in kernels)
    return (end_ns - start_ns) / 1000.0


def _kernel_busy_us(kernels: list[dict]) -> float:
    """Total time the device spent executing these kernels, overlaps counted once."""
    if not kernels:
        return 0.0
    intervals = sorted((int(k["start_ns"]), int(k["end_ns"])) for k in kernels)
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return (total + current_end - current_start) / 1000.0


def _cuda_events_fallback_enabled() -> bool:
    return os.getenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0") == "1"


def _attributed_samples(
    kernels: list[dict],
    iteration_of: dict[int, int],
    n_repeat: int,
    dropped: Optional[int] = 0,
    count_copies: bool = False,
) -> list[Sample]:
    """Return one Sample per iteration.

    A kernel belongs to the iteration whose correlation id it carries, so a call whose
    kernel count varies between iterations is measured rather than rejected. Both ways
    of coming up short -- an unclaimed kernel, an iteration with none -- can also mean
    the record was taken and then discarded, which is what ``dropped`` distinguishes.

    A device-to-device copy the call issued is device work like any other. Where the case
    does not count copies it is left out of ``device_busy_ms`` and reported in
    ``uncounted_copy_ms``, so a call that computes part of its result with a copy is
    visibly reading faster than it is rather than silently.
    """
    claimed: dict[int, list[dict]] = {}
    orphans = []
    for kernel in kernels:
        iteration = iteration_of.get(kernel["correlation_id"])
        if iteration == _PREPARE_ID:
            continue
        if iteration is None or not 0 <= iteration < n_repeat:
            # An id this phase never pushed is as unattributable as no id at all.
            orphans.append(kernel)
        else:
            claimed.setdefault(iteration, []).append(kernel)

    uncounted: dict[int, float] = {}
    if not count_copies:
        for iteration, group in list(claimed.items()):
            copies = [k for k in group if k.get("is_copy")]
            if not copies:
                continue
            uncounted[iteration] = _kernel_busy_us(copies) * 1e-3
            kept = [k for k in group if not k.get("is_copy")]
            # An iteration that only copied has nothing left to time.
            if kept:
                claimed[iteration] = kept
            else:
                del claimed[iteration]

    unmeasured = [i for i in range(n_repeat) if i not in claimed]
    if orphans or unmeasured:
        short = (
            f"{len(unmeasured)} of {n_repeat} iterations have no kernel of their "
            f"own and {len(orphans)} kernels belong to no iteration"
        )
        if dropped:
            raise _CUPTIRecordsLostError(
                f"{short}; CUPTI discarded {dropped} activity records for want of "
                f"buffer space, so those readings were lost rather than never taken"
            )
        if dropped is None:
            raise _CUPTIAttributionError(
                f"{short}; CUPTI's dropped-record count is unproven, so a lost record "
                f"cannot be ruled out"
            )
        if orphans:
            raise _OffThreadLaunchError(
                f"{len(orphans)} of {len(kernels)} kernels carry no iteration id "
                f"(first: {orphans[0]['name']}); the call handed its work to a thread "
                f"the id was not pushed on"
            )
        raise _CUPTIAttributionError(
            f"{len(unmeasured)} of {n_repeat} iterations launched no kernel "
            f"(first: iteration {unmeasured[0]}); CUPTI discarded nothing, so a call "
            f"that never reaches the device is not being measured"
        )

    return [
        Sample(
            device_busy_ms=_kernel_busy_us(claimed[i]) * 1e-3,
            latency_ms=_kernel_span_us(claimed[i]) * 1e-3,
            n_kernels=len(claimed[i]),
            uncounted_copy_ms=uncounted.get(i, 0.0),
        )
        for i in range(n_repeat)
    ]


def _collect_attributed(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None],
    count_copies: bool = False,
) -> list[Sample]:
    """Collect one fully attributed phase, re-measuring what CUPTI loses.

    A discarded record is the instrument failing rather than the op: the iteration ran
    and only its reading is gone, so the phase is run again, each attempt asking for a
    larger buffer. Anything the drop counter does not explain fails on the first attempt.
    """
    buffer_bytes = _BUFFER_BYTES
    for attempt in range(_ATTRIBUTION_ATTEMPTS):
        trace = collect_repeats(run_one, n_repeat, prepare_one, buffer_bytes)
        try:
            return _attributed_samples(
                trace.kernels,
                trace.iteration_of,
                n_repeat,
                trace.dropped,
                count_copies,
            )
        except _CUPTIRecordsLostError as exc:
            _bench_meta.attribution_retries = attempt + 1
            if attempt == _ATTRIBUTION_ATTEMPTS - 1:
                raise
            buffer_bytes *= 4
            _logger.warning(
                "%s; re-measuring with a %d KB buffer (attempt %d of %d).",
                exc,
                buffer_bytes // 1024,
                attempt + 2,
                _ATTRIBUTION_ATTEMPTS,
            )
    raise AssertionError("the loop returns or raises on its last attempt")


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
                "L2 cache size query returned %d; flushing a 256 MB buffer instead",
                l2_bytes,
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


def _capture_bench_meta() -> dict:
    """Snapshot how the last measurement was taken."""
    return {
        key: value
        for key in ("timing", "fallback_reason", "attribution_retries")
        if (value := getattr(_bench_meta, key, None)) is not None
    }


def median_busy_ms(samples: "list[Sample]") -> float:
    """The device-busy figure a report publishes for a run: the median sample.

    One definition, so a baseline that ranks candidates on this number and the
    row that reports it cannot drift apart.
    """
    return statistics.median(s.device_busy_ms for s in samples)


def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    dry_run_ms: float = DRY_RUN_MS,
    repeat_ms: float = REPEAT_MS,
    max_iters: int = _MAX_ITERS,
    min_iters: int = _MIN_ITERS,
    count_copies: bool = False,
) -> list[Sample]:
    """Time *fn* through CUPTI, one :class:`Sample` per iteration.

    A calibration pass measures one cold-isolated iteration and the millisecond budgets
    divide into it, clamped to ``[min_iters, max_iters]``; an op faster than the L2 flush
    takes its count from the clamp. L2 is cleared before every iteration, and each call
    is drained before the next begins. Each kernel is attributed to the iteration that
    launched it, so *fn* must launch its own work rather than hand it to another thread.
    A phase whose records CUPTI discarded is measured again; attribution otherwise
    fails closed unless
    ``TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1``.

    ``count_copies`` adds device-to-device copies to what is attributed, for a row whose
    implementations compute part of the result with one. It is off by default because a
    staging copy is not the arithmetic being timed; turn it on for every tag in a row or
    for none, since the two sides are otherwise read off different instruments.
    """
    if not isinstance(args, tuple):
        raise TypeError(
            f"bench_kernel expects a tuple of args, got {type(args).__name__}. "
            "Check that gen_inputs() returns a tuple."
        )

    allow_fallback = _cuda_events_fallback_enabled()
    _bench_meta.timing = None
    _bench_meta.fallback_reason = None
    _bench_meta.attribution_retries = None
    cache = _get_l2_flush_cache()

    def _flush_l2():
        _reset_persisting_l2_cache()
        cache.zero_()

    def _call_raw():
        return fn(*args) if args else fn()

    _flush_l2()
    _call_raw()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_CALIBRATION_ITERS):
        _flush_l2()
        _call_raw()
    end.record()
    torch.cuda.synchronize()
    per_iter_ms = max(start.elapsed_time(end) / _CALIBRATION_ITERS, 1e-6)

    n_warmup = _clamp_iters(dry_run_ms / per_iter_ms, max_iters, min_iters)
    n_repeat = _clamp_iters(repeat_ms / per_iter_ms, max_iters, min_iters)

    def _run(i):
        return fn(*args) if args else fn()

    def _prepare_iteration(i):
        _flush_l2()
        torch.cuda.synchronize()

    for i in range(n_warmup):
        _prepare_iteration(i)
        _run(i)
    torch.cuda.synchronize()

    try:
        with _native_output_suppressor():
            samples = _collect_attributed(_run, n_repeat, _prepare_iteration, count_copies)
        _bench_meta.timing = "cupti"
    except (_CUPTIAttributionError, CUPTIError) as exc:
        if not allow_fallback:
            raise RuntimeError(
                f"CUPTI profiling failed: {exc}. CUDA-events fallback is disabled "
                "(TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0), which keeps the run from "
                "silently mixing two timing methods."
            ) from exc
        _bench_meta.timing = "cuda-events"
        _bench_meta.fallback_reason = str(exc)
        _logger.warning("CUPTI timing failed (%s); falling back to CUDA events.", exc)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        for i in range(n_repeat):
            _prepare_iteration(i)
            starts[i].record()
            _run(i)
            ends[i].record()
        torch.cuda.synchronize()
        # Events bracket the call, so they cannot separate execution from the gaps
        # between kernels. Both fields carry the same number and `timing` says why.
        samples = [
            Sample(device_busy_ms=elapsed, latency_ms=elapsed, n_kernels=None)
            for elapsed in (s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
        ]

    torch.cuda.empty_cache()
    return samples
