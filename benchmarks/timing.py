"""How a call is timed: the loop around CUPTI activity collection (``tileops._utils.cupti``).

The timing layer owns the measurement and nothing else -- it knows how to run a callable
n times and return each run's device latency. What the numbers mean, and where they are
written, belong to the layers above.

Every iteration restores the mutable inputs, reads a private eviction buffer, then calls;
only the call is read. Three prepared calls calibrate the warm-up and repeat counts.

A CUDA device is timed with the pieces of NVIDIA nvbench's cold measurement
(``nvbench/detail/measure_cold.cuh``): every sample flushes L2 before the call, runs behind
nvbench's blocking kernel, and records SM 0's clock around the call; persisting L2 priority
is reset before each block of samples. nvbench holds each sample alone and drains the
device after it; here one blocking kernel holds a block of samples queued with no host
synchronization between them, so host launch cost never becomes device idle time. Warm-up
loads the blocking kernel, then runs each call unheld and drained, as nvbench's does. No
sample is discarded for its clock (nvbench's ``--throttle-threshold 0``): a compute-bound
call at the power cap runs throttled in steady state, and the protocol asks only that every
implementation run in the same power state. A block whose queue filled
before the host released it is measured again in smaller blocks; where the queue holds no
sample, every sample is measured again unheld and drained, as nvbench runs a launcher that
cannot be held (``exec_tag::sync``). ``block_samples`` records the most samples a block
then held, 0 for none, and ``l2_flush`` whether the samples were held or drained.

The readings come from CUPTI. Another accelerator is timed with torch's device events,
which bracket each call: its ``timing`` reads ``events``, and the busy time equals the
latency.

A kernel belongs to the iteration whose external correlation id it carries, so nothing
is inferred from timestamps and a kernel shorter than the host overhead around it is
attributed as reliably as a long one. CUPTI's id stack is per-thread, which is this
protocol's one demand on a timed callable: it must launch its own work rather than hand
it to another thread (``Tensor.backward`` hands it to autograd's engine thread;
``grad_fn.apply`` does not).
"""

import contextlib
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

import torch

from tileops._utils.cupti import (
    AttributionError,
    CUPTIError,
    Sample,
    collect_attributed,
    l2_flush_buffer,
    reset_persisting_l2_cache,
)
from tileops._utils.device import device_context, on_current_index
from workloads.device import run_device

_logger = logging.getLogger("tileops.bench")

# Per-phase wall-time budgets in ms, divided by the cost of one iteration: reset, L2
# flush and call. That cost includes the flush, which the reading excludes and which on
# its own outweighs a microsecond kernel, so most ops take their count from _MAX_ITERS.
DRY_RUN_MS = 25.0
REPEAT_MS = 100.0
# Counts are clamped: attribution requires every repeat to reach the GPU, so an
# unbounded count turns one hiccup into a failed case.
_MIN_ITERS = 10
_MAX_ITERS = 200

# How long the blocking kernel holds the stream before releasing it on its own. A queue too
# full to take the next launch stalls the host until then, so this is the cost of finding
# that a block holds too many samples; nvbench's 30 s treats the same event as a failure.
_BLOCK_TIMEOUT_S = 1.0

# Latest bench_kernel measurement metadata; deviations from the default protocol are
# surfaced in every recorded result.
_bench_meta = threading.local()


# Cache preparation identifiers, recorded with each result, keep results taken under
# different preparations apart. On CUDA a sample is either queued behind the blocking
# kernel or, where none can be held, drained before and after its call; elsewhere samples
# are queued with no drain between them.
_L2_FLUSH_CUDA = "triton-read-2x-l2-held"
_L2_FLUSH_CUDA_DRAINED = "triton-read-2x-l2-drained"
_L2_FLUSH_OTHER = "torch-max-2x-l2-undrained"


def _clamp_iters(raw: float, max_iters: int = _MAX_ITERS, min_iters: int = _MIN_ITERS) -> int:
    return max(min_iters, min(max_iters, int(raw)))


# Whether this run may time with CUDA events where CUPTI cannot attribute a phase.
# The run sets it (``--tileops-allow-events-fallback`` in ``benchmarks/conftest.py``).
_events_fallback_allowed = False


def events_fallback_allowed() -> bool:
    """Whether this run lets ``bench_kernel`` fall back to CUDA events; ``False`` unless set."""
    return _events_fallback_allowed


def set_events_fallback_allowed(allowed: bool) -> None:
    """Let ``bench_kernel`` fall back to CUDA events from now on, or refuse to."""
    global _events_fallback_allowed
    _events_fallback_allowed = allowed


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


def bench_device() -> torch.device:
    """The device this run times on: ``--tileops-device``, with its index filled in."""
    return on_current_index(torch.device(run_device()))


def _instrument_for(device: torch.device) -> str:
    """How a call on *device* is timed: ``cupti`` on CUDA, ``events`` on any other device."""
    return "cupti" if device.type == "cuda" else "events"


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
        for key in (
            "timing",
            "fallback_reason",
            "attribution_retries",
            "l2_flush",
            "block_samples",
        )
        if (value := getattr(_bench_meta, key, None)) is not None
    }


def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    dry_run_ms: float = DRY_RUN_MS,
    repeat_ms: float = REPEAT_MS,
    max_iters: int = _MAX_ITERS,
    min_iters: int = _MIN_ITERS,
    count_copies: bool = False,
    allow_events_fallback: bool | None = None,
    reset: Callable[[], None] | None = None,
    device: torch.device | None = None,
) -> list[Sample]:
    """Time *fn* on *device*, one :class:`Sample` per iteration, by the protocol in this
    module's docstring.

    The warm-up and repeat counts are *dry_run_ms* and *repeat_ms* divided by the calibrated
    cost of one iteration, clamped to ``[min_iters, max_iters]``. A phase whose
    records CUPTI discarded is measured again; attribution otherwise fails closed unless
    ``allow_events_fallback`` is true. ``None`` takes the run's setting,
    ``--tileops-allow-events-fallback``.

    ``count_copies`` adds device-to-device copies to what is attributed, for a row whose
    implementations compute part of the result with one. It is off by default because a
    staging copy is not the arithmetic being timed; turn it on for every tag in a row or
    for none, since the two sides are otherwise read off different instruments.

    ``reset`` restores the state *fn* overwrites, before the L2 flush, which evicts what it
    wrote. It enqueues its work on the current stream, the flush's, which orders the two.

    Everything runs on *device*, by default :func:`bench_device`. A CUDA device is timed
    through CUPTI; any other is timed with device events, which neither attribute kernels
    nor fall back.

    Raises:
        TypeError: *args* is not a tuple.
        ValueError: A tensor in *args* is on another device than *device*, CPU aside.
    """
    if not isinstance(args, tuple):
        raise TypeError(
            f"bench_kernel expects a tuple of args, got {type(args).__name__}. "
            "Check that gen_inputs() returns a tuple."
        )
    device = bench_device() if device is None else on_current_index(device)
    elsewhere = {
        a.device for a in args if isinstance(a, torch.Tensor) and a.device.type != "cpu"
    } - {device}
    if elsewhere:
        raise ValueError(
            f"bench_kernel times on {device}, but its args hold tensors on "
            f"{sorted(map(str, elsewhere))}"
        )
    with device_context(device):
        return _bench_kernel(
            device,
            fn,
            args,
            dry_run_ms,
            repeat_ms,
            max_iters,
            min_iters,
            count_copies,
            allow_events_fallback,
            reset,
        )


# One blocker per CUDA device: its flags are pinned host memory, allocated once.
_BLOCKERS: dict[torch.device, Any] = {}


def _bench_kernel(
    device: torch.device,
    fn: Callable,
    args: tuple[Any, ...] = (),
    dry_run_ms: float = DRY_RUN_MS,
    repeat_ms: float = REPEAT_MS,
    max_iters: int = _MAX_ITERS,
    min_iters: int = _MIN_ITERS,
    count_copies: bool = False,
    allow_events_fallback: bool | None = None,
    reset: Callable[[], None] | None = None,
) -> list[Sample]:
    allow_fallback = (
        events_fallback_allowed() if allow_events_fallback is None else allow_events_fallback
    )
    _bench_meta.timing = None
    _bench_meta.fallback_reason = None
    _bench_meta.attribution_retries = None
    _bench_meta.block_samples = None
    _bench_meta.l2_flush = None
    cache = l2_flush_buffer(device)
    on_cuda = device.type == "cuda"

    def _reset_and_flush():
        if reset is not None:
            reset()
        if on_cuda:
            read_sweep(cache)
        else:
            cache.max()

    def _call_raw(_iteration: int = 0):
        return fn(*args)

    if on_cuda:
        # Imported on first use: Triton is needed only where a CUDA device is timed.
        from tileops._utils._l2_sweep import read_sweep

        reset_persisting_l2_cache()
    _reset_and_flush()
    _call_raw()
    torch.accelerator.synchronize()
    calibration_iters = 3
    start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
    start.record()
    for _ in range(calibration_iters):
        _reset_and_flush()
        _call_raw()
    end.record()
    torch.accelerator.synchronize()
    per_iter_ms = max(start.elapsed_time(end) / calibration_iters, 1e-6)

    n_warmup = _clamp_iters(dry_run_ms / per_iter_ms, max_iters, min_iters)
    n_repeat = _clamp_iters(repeat_ms / per_iter_ms, max_iters, min_iters)

    if _instrument_for(device) == "events":
        for _ in range(n_warmup):
            _reset_and_flush()
            _call_raw()
        torch.accelerator.synchronize()
        _bench_meta.timing = "events"
        _bench_meta.l2_flush = _L2_FLUSH_OTHER
        samples = _event_samples(_call_raw, n_repeat, lambda i: _reset_and_flush())
        torch.accelerator.empty_cache()
        return samples

    from benchmarks.cold import SmClocks, StreamBlocker

    blocker = _BLOCKERS.get(device)
    if blocker is None:
        blocker = _BLOCKERS[device] = StreamBlocker(_BLOCK_TIMEOUT_S)
    clocks = SmClocks(n_repeat, device)
    # The samples the current block holds; whether the blocker holds them; how many were
    # fully queued before the blocker released on its own, None while it held; and the
    # attempts the current block's reading lost records on.
    size, blocking, fitted, retried = 0, False, None, 0

    def _prepare(i):
        nonlocal fitted
        if i == 0:
            fitted = None
            clocks.clear()
            # Before the blocker, not per sample: the reset takes effect when the host calls
            # it, so queued samples share one either way, and under CUPTI a reset issued
            # while the stream is held stalls a later launch until the blocker times out.
            reset_persisting_l2_cache()
            if blocking:
                blocker.block()
        _reset_and_flush()
        # A sample the blocker does not hold is nvbench's cold sample as its timer takes it
        # (measure_cold_launch_timer_core): drained after the flush and after the call.
        if not blocking:
            torch.accelerator.synchronize()
        clocks.start(i)

    def _finish(i):
        nonlocal fitted
        clocks.stop(i)
        if not blocking:
            torch.accelerator.synchronize()
            return
        if fitted is None and blocker.timed_out():
            fitted = i
        if i == size - 1:
            blocker.unblock()

    def _count_retries(attempts: int) -> None:
        nonlocal retried
        retried = attempts

    def _read(reading: Callable[[int], list[Sample]]) -> list[Sample]:
        """Take *n_repeat* samples in blocks held by the blocking kernel, or all of them
        unheld where the queue holds no sample."""
        nonlocal size, blocking, retried
        blocking, per_block = True, n_repeat
        _bench_meta.block_samples = None
        _bench_meta.l2_flush = _L2_FLUSH_CUDA
        # The row's retries are those of the blocks it keeps.
        _bench_meta.attribution_retries = None
        kept: list[Sample] = []
        while len(kept) < n_repeat:
            size = min(per_block, n_repeat - len(kept))
            retried = 0
            readings = reading(size)
            if blocking and blocker.timed_out():
                # The queue took fewer samples than the block held. A queue that took none
                # drops what was kept, so no reading mixes held and drained samples.
                per_block = fitted if fitted is not None else size - 1
                if per_block == 0:
                    blocking, per_block = False, n_repeat
                    kept.clear()
                    _bench_meta.l2_flush = _L2_FLUSH_CUDA_DRAINED
                    _bench_meta.attribution_retries = None
                _bench_meta.block_samples = per_block if blocking else 0
                continue
            kept.extend(
                sample._replace(sm_clock_hz=hz)
                for sample, hz in zip(readings, clocks.read_hz(size), strict=True)
            )
            if retried:
                _bench_meta.attribution_retries = (_bench_meta.attribution_retries or 0) + retried
        return kept

    try:
        # nvbench loads the blocking kernel before warm-up (NVIDIA/nvbench#339), then warms
        # up without it (NVIDIA/nvbench#240): a kernel loaded lazily while the stream is
        # blocked stalls the host until the blocker times out.
        blocker.block()
        blocker.unblock()
        for i in range(n_warmup):
            _prepare(i % n_repeat)
            _call_raw()
            _finish(i % n_repeat)
        try:
            with _native_output_suppressor():
                samples = _read(
                    lambda n: collect_attributed(
                        _call_raw,
                        n,
                        _prepare,
                        count_copies,
                        on_retry=_count_retries,
                        # A call that lost one of its kernels to a full buffer reads short.
                        every_record=True,
                        finish_one=_finish,
                    )
                )
            _bench_meta.timing = "cupti"
        except (AttributionError, CUPTIError) as exc:
            if not allow_fallback:
                raise RuntimeError(
                    f"CUPTI profiling failed: {exc}. CUDA-events fallback is disabled "
                    "(pass --tileops-allow-events-fallback to allow it), which keeps the run "
                    "from silently mixing two timing methods."
                ) from exc
            _bench_meta.timing = "cuda-events"
            _bench_meta.fallback_reason = str(exc)
            _logger.warning("CUPTI timing failed (%s); falling back to CUDA events.", exc)
            samples = _read(lambda n: _event_samples(_call_raw, n, _prepare, _finish))
    finally:
        # A call that raised while queued behind the blocker must not leave the stream held.
        blocker.unblock()

    torch.accelerator.empty_cache()
    return samples


def _event_samples(
    run: Callable[[int], Any],
    n_repeat: int,
    prepare: Callable[[int], None],
    finish: Callable[[int], None] | None = None,
) -> list[Sample]:
    """Time *n_repeat* iterations of *run* with device events on the current device.

    Events bracket the call, so they cannot separate execution from the gaps between
    kernels, nor a device-to-device copy from the arithmetic. Both fields carry the same
    number and ``timing`` says why.
    """
    starts = [torch.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [torch.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        prepare(i)
        starts[i].record()
        run(i)
        ends[i].record()
        if finish is not None:
            finish(i)
    torch.accelerator.synchronize()
    return [
        Sample(device_busy_ms=elapsed, latency_ms=elapsed, n_kernels=None)
        for elapsed in (s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    ]
