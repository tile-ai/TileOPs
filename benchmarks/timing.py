"""How a call is timed: the loop around CUPTI activity collection (``tileops._utils.cupti``).

The timing layer owns the measurement and nothing else -- it knows how to run a callable
n times and return each run's device latency. What the numbers mean, and where they are
written, belong to the layers above.

A CUDA device is timed through CUPTI. Another accelerator is timed with torch's device
events, which bracket each call: its ``timing`` reads ``events``, and the busy time equals
the latency.

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
from typing import Any, Callable, Optional

import torch

from tileops._utils.cupti import (
    AttributionError,
    CUPTIError,
    Sample,
    collect_attributed,
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

# Latest bench_kernel measurement metadata; deviations from the default protocol are
# surfaced in every recorded result.
_bench_meta = threading.local()


# L2 cache flush buffer per device, allocated lazily. Keyed by the whole device: two
# device types both number their first device 0.
_l2_flush_caches: dict[torch.device, torch.Tensor] = {}
# Cache preparation identifiers separate incompatible historical measurements.
_L2_FLUSH_CUDA = "triton-read-2x-l2"
_L2_FLUSH_OTHER = "torch-max-2x-l2"
# The L2 assumed for a device whose torch module reports none.
_FALLBACK_L2_BYTES = int(256e6)


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


def _read_sweep(buffer: torch.Tensor) -> None:
    # Imported on first use: Triton is needed only where a CUDA device is timed.
    from benchmarks._l2_sweep import read_sweep

    read_sweep(buffer)


def bench_device() -> torch.device:
    """The device this run times on: ``--tileops-device``, with its index filled in."""
    return on_current_index(torch.device(run_device()))


def _instrument_for(device: torch.device) -> str:
    """How a call on *device* is timed: ``cupti`` on CUDA, ``events`` on any other device."""
    return "cupti" if device.type == "cuda" else "events"


def _get_l2_flush_cache(device: torch.device) -> torch.Tensor:
    """Cache a private int32 buffer covering twice the device's L2 capacity.

    Read-only eviction avoids buffer-sized dirty writebacks during the timed call.
    Private CPU-generated random values preserve benchmark RNG state and resist memory
    compression. Missing capacity reports use ``_FALLBACK_L2_BYTES``.
    """
    cache = _l2_flush_caches.get(device)
    if cache is None:
        module = torch.get_device_module(device.type)
        properties = getattr(module, "get_device_properties", None)
        l2_bytes = getattr(properties(device.index), "L2_cache_size", 0) if properties else 0
        reported = l2_bytes
        if l2_bytes <= 0:
            l2_bytes = _FALLBACK_L2_BYTES
        values = torch.randint(
            -(2**31),
            2**31 - 1,
            (-(-2 * l2_bytes // 4),),
            dtype=torch.int32,
            device="cpu",
            generator=torch.Generator().manual_seed(0),
        )
        cache = values.to(device)
        if reported <= 0:
            _logger.warning(
                "L2 cache size query on %s returned %d; assuming %d bytes of L2, so each "
                "flush reads %d bytes",
                device,
                reported,
                l2_bytes,
                cache.numel() * cache.element_size(),
            )
        _l2_flush_caches[device] = cache
    return cache


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
        for key in ("timing", "fallback_reason", "attribution_retries", "l2_flush")
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
    allow_events_fallback: Optional[bool] = None,
    reset: Optional[Callable[[], None]] = None,
    device: Optional[torch.device] = None,
) -> list[Sample]:
    """Time *fn* on *device*, one :class:`Sample` per iteration.

    Every call uses the same preparation: restore mutable inputs, reset persisting L2
    priority on CUDA, then read a private eviction buffer. Preparation is excluded from
    timing. Warm-up and timed calls drain the device after preparation; calibration
    measures three consecutive prepared calls to set the iteration counts within
    ``[min_iters, max_iters]``. Each kernel is attributed to the iteration that launched
    it, so *fn* must launch its own work rather than hand it to another thread.
    A phase whose records CUPTI discarded is measured again; attribution otherwise
    fails closed unless ``allow_events_fallback`` is true. ``None`` takes the run's
    setting, ``--tileops-allow-events-fallback``.

    ``count_copies`` adds device-to-device copies to what is attributed, for a row whose
    implementations compute part of the result with one. It is off by default because a
    staging copy is not the arithmetic being timed; turn it on for every tag in a row or
    for none, since the two sides are otherwise read off different instruments.

    ``reset`` restores the state *fn* overwrites. Every iteration runs ``reset``, then the
    L2 flush, then *fn*; ``reset`` is never part of the reading, and the flush evicts what
    it wrote. ``reset`` enqueues its work on the current stream, the flush's, which is what
    orders the two.

    Everything runs on *device*, by default :func:`bench_device`. A CUDA device is timed
    through CUPTI as above; any other is timed with device events, which neither attribute
    kernels nor fall back.

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


def _bench_kernel(
    device: torch.device,
    fn: Callable,
    args: tuple[Any, ...] = (),
    dry_run_ms: float = DRY_RUN_MS,
    repeat_ms: float = REPEAT_MS,
    max_iters: int = _MAX_ITERS,
    min_iters: int = _MIN_ITERS,
    count_copies: bool = False,
    allow_events_fallback: Optional[bool] = None,
    reset: Optional[Callable[[], None]] = None,
) -> list[Sample]:
    allow_fallback = (
        events_fallback_allowed() if allow_events_fallback is None else allow_events_fallback
    )
    _bench_meta.timing = None
    _bench_meta.fallback_reason = None
    _bench_meta.attribution_retries = None
    cache = _get_l2_flush_cache(device)
    on_cuda = device.type == "cuda"
    _bench_meta.l2_flush = _L2_FLUSH_CUDA if on_cuda else _L2_FLUSH_OTHER

    def _reset_and_flush():
        if reset is not None:
            reset()
        if on_cuda:
            reset_persisting_l2_cache()
            _read_sweep(cache)
        else:
            cache.max()

    def _call_raw():
        return fn(*args) if args else fn()

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

    def _run(i):
        return fn(*args) if args else fn()

    def _prepare_iteration(i):
        _reset_and_flush()
        torch.accelerator.synchronize()

    for i in range(n_warmup):
        _prepare_iteration(i)
        _run(i)
    torch.accelerator.synchronize()

    if _instrument_for(device) == "events":
        _bench_meta.timing = "events"
        samples = _event_samples(_run, n_repeat, _prepare_iteration)
        torch.accelerator.empty_cache()
        return samples
    try:
        with _native_output_suppressor():
            samples = collect_attributed(
                _run,
                n_repeat,
                _prepare_iteration,
                count_copies,
                on_retry=lambda attempts: setattr(_bench_meta, "attribution_retries", attempts),
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
        samples = _event_samples(_run, n_repeat, _prepare_iteration)

    torch.accelerator.empty_cache()
    return samples


def _event_samples(
    run: Callable[[int], Any], n_repeat: int, prepare: Callable[[int], None]
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
    torch.accelerator.synchronize()
    return [
        Sample(device_busy_ms=elapsed, latency_ms=elapsed, n_kernels=None)
        for elapsed in (s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    ]
