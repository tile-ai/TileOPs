"""CUPTI kernel activity: one collector per process, and the device-busy reading it yields.

CUPTI delivers activity records to one pair of buffer callbacks per process, so every
reading in a process goes through this collector: the benchmarks' timer and the autotuner
alike. A reading is one :class:`Sample` per iteration, its ``device_busy_ms`` the union of
the kernel intervals the iteration launched; work run between iterations is excluded.

A kernel belongs to the iteration whose external correlation id it carries, so nothing
is inferred from timestamps. CUPTI's id stack is per-thread: a timed callable must launch
its own work rather than hand it to another thread.
"""

import contextlib
import ctypes
import logging
import statistics
from typing import Any, Callable, NamedTuple, Optional

import torch

__all__ = [
    "AttributionError",
    "CUPTIError",
    "OffThreadLaunchError",
    "RecordsLostError",
    "Sample",
    "Trace",
    "attributed_samples",
    "collect_attributed",
    "collect_repeats",
    "flush_l2",
    "l2_flush_buffer",
    "median_busy_ms",
    "reset_persisting_l2_cache",
]

_logger = logging.getLogger(__name__)

# CUPTI activity collection, via NVIDIA's cupti-python binding.

_CUPTI = None
_COLLECTOR_ACTIVE = False
_CALLBACKS_REGISTERED = False
# Whatever CUPTI does with a buffer between handing it back and asking for the next one
# scales with its size and runs on this thread, inside a timed call, so a buffer of a
# few megabytes stalls iterations of a short call. Only latency_ms picks the stall up;
# the records are the same either way.
_BUFFER_BYTES = 256 * 1024
# What the next buffer request answers with. A phase that lost records raises it for
# the retry: a stall costs one iteration's latency_ms, a lost record costs the case.
_buffer_bytes = _BUFFER_BYTES
_KERNELS: list[dict[str, Any]] = []
_ITERATION_OF: dict[int, int] = {}
_DROPPED_COUNT_UNREADABLE = False
_DROP_COUNTER_LIVE: Optional[bool] = None
# Pushed around the L2 flush and anything else run between iterations. Labelling that
# work is what lets a kernel with no id at all mean one thing only: a thread the id was
# never pushed on launched it. Out of range of any iteration index.
_PREPARE_ID = 1 << 32
_cuda_runtime = None
# L2 flush buffer per device, allocated lazily. Keyed by the whole device: two device types
# both number their first device 0.
_L2_FLUSH_BUFFERS: dict[torch.device, torch.Tensor] = {}
# The L2 assumed for a device whose torch module reports none.
_FALLBACK_L2_BYTES = int(256e6)


class AttributionError(Exception):
    """A CUPTI trace could not be attributed to logical benchmark calls."""


class RecordsLostError(AttributionError):
    """A reading was taken and then discarded. Its own class because it can be retried."""


class OffThreadLaunchError(AttributionError):
    """Kernels ran that no iteration claims, so a thread without the id launched them."""


class CUPTIError(RuntimeError):
    """The CUPTI collector is unavailable or could not be operated."""


def _load_cupti():
    global _CUPTI
    if _CUPTI is not None:
        return _CUPTI
    try:
        from cupti import cupti
    except Exception as exc:
        raise CUPTIError(
            "cupti-python is unavailable. Install it with "
            "`pip install --no-deps cupti-python==13.2.0`; --no-deps is required "
            "or it downgrades torch's cuda-bindings pin."
        ) from exc
    _CUPTI = cupti
    return _CUPTI


def _buffer_requested():
    buffer_alignment = 8
    return _buffer_bytes, buffer_alignment


def _read_dropped() -> Optional[int]:
    """Read CUPTI's drop counter, which resets on every read. None if the call fails."""
    global _DROPPED_COUNT_UNREADABLE
    cupti = _load_cupti()
    dropped = ctypes.c_size_t(0)
    try:
        # Kernel records land on the global queue, which is context 0, and the binding
        # takes the out-parameter as an address rather than returning it.
        cupti.activity_get_num_dropped_records(0, 0, ctypes.addressof(dropped))
    except Exception as exc:
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
    """Validate the drop counter once per process using a deliberately undersized buffer."""
    global _DROP_COUNTER_LIVE
    if _DROP_COUNTER_LIVE is not None:
        return _DROP_COUNTER_LIVE
    probe_kernels = 4
    probe = torch.empty(1, device="cuda")
    with _phase_session(buffer_bytes=8):  # too small to hold one record
        for _ in range(probe_kernels):
            probe.zero_()
        torch.cuda.synchronize()  # lint: device-api allow: CUPTI runs on CUDA only
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
    """Trace launch and memcpy APIs for correlation, after enabling activity collection."""
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
            except Exception:
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
    except Exception as exc:
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
        except Exception as exc:
            raise CUPTIError(f"CUPTI collector failed to stop: {exc}") from exc


def _flush() -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Return the kernels recorded since the previous flush, and their iteration map."""
    cupti = _load_cupti()
    torch.cuda.synchronize()  # lint: device-api allow: CUPTI runs on CUDA only
    try:
        cupti.activity_flush_all(1)  # CUPTI_ACTIVITY_FLAG_FLUSH_FORCED
    except Exception as exc:
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
            torch.cuda.synchronize()  # lint: device-api allow: CUPTI runs on CUDA only
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


def attributed_samples(
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
            raise RecordsLostError(
                f"{short}; CUPTI discarded {dropped} activity records for want of "
                f"buffer space, so those readings were lost rather than never taken"
            )
        if dropped is None:
            raise AttributionError(
                f"{short}; CUPTI's dropped-record count is unproven, so a lost record "
                f"cannot be ruled out"
            )
        if orphans:
            raise OffThreadLaunchError(
                f"{len(orphans)} of {len(kernels)} kernels carry no iteration id "
                f"(first: {orphans[0]['name']}); the call handed its work to a thread "
                f"the id was not pushed on"
            )
        raise AttributionError(
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


def collect_attributed(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None],
    count_copies: bool = False,
    on_retry: Optional[Callable[[int], None]] = None,
    every_record: bool = False,
) -> list[Sample]:
    """Collect one fully attributed phase, re-measuring what CUPTI loses.

    A discarded record is the instrument failing rather than the op: the iteration ran
    and only its reading is gone, so the phase is run again, each attempt asking for a
    larger buffer. Anything the drop counter does not explain fails on the first attempt.
    ``on_retry`` receives the number of attempts that lost records, after each one.

    ``every_record`` treats any record CUPTI reports discarded, and a drop count it could not
    read, as a lost reading, even one whose iterations all kept a kernel: a call that lost
    one of its kernels reads short. A drop counter the process proved dead fails on the first
    attempt, as no buffer size makes it count.
    """
    attribution_attempts = 3
    buffer_bytes = _BUFFER_BYTES
    for attempt in range(attribution_attempts):
        trace = collect_repeats(run_one, n_repeat, prepare_one, buffer_bytes)
        try:
            if every_record and trace.dropped is None and not _drop_counter_is_live():
                raise AttributionError(
                    "CUPTI's dropped-record counter is dead in this process, so a call that "
                    "lost one of its kernels cannot be told from one that launched no more"
                )
            if every_record and trace.dropped != 0:
                lost = "an unproven number of" if trace.dropped is None else trace.dropped
                raise RecordsLostError(
                    f"CUPTI discarded {lost} activity records, so a call may read "
                    f"short of the kernels it launched"
                )
            return attributed_samples(
                trace.kernels,
                trace.iteration_of,
                n_repeat,
                trace.dropped,
                count_copies,
            )
        except RecordsLostError as exc:
            if on_retry is not None:
                on_retry(attempt + 1)
            if attempt == attribution_attempts - 1:
                raise
            buffer_bytes *= 4
            _logger.warning(
                "%s; re-measuring with a %d KB buffer (attempt %d of %d).",
                exc,
                buffer_bytes // 1024,
                attempt + 2,
                attribution_attempts,
            )
    raise AssertionError("the loop returns or raises on its last attempt")


def reset_persisting_l2_cache() -> None:
    global _cuda_runtime
    if _cuda_runtime is None:
        from cuda.bindings import runtime as cuda_runtime

        _cuda_runtime = cuda_runtime

    result = _cuda_runtime.cudaCtxResetPersistingL2Cache()
    if isinstance(result, tuple):
        result = result[0]
    torch.cuda.check_error(int(result))  # lint: device-api allow: CUPTI runs on CUDA only


def l2_flush_buffer(device: torch.device) -> torch.Tensor:
    """A private int32 buffer on *device* covering twice its L2 capacity, drawn once.

    Private CPU-generated random values preserve the caller's RNG state and resist memory
    compression. Missing capacity reports use ``_FALLBACK_L2_BYTES``.
    """
    buffer = _L2_FLUSH_BUFFERS.get(device)
    if buffer is None:
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
        buffer = values.to(device)
        if reported <= 0:
            _logger.warning(
                "L2 cache size query on %s returned %d; assuming %d bytes of L2, so each "
                "flush reads %d bytes",
                device,
                reported,
                l2_bytes,
                buffer.numel() * buffer.element_size(),
            )
        _L2_FLUSH_BUFFERS[device] = buffer
    return buffer


def flush_l2(buffer: torch.Tensor) -> None:
    """Return persisting L2 lines to normal priority, then evict L2 by reading *buffer*, an
    :func:`l2_flush_buffer` on a CUDA device. The preparation before every call that
    benchmarks and tuning read, so both rank candidates from the same cache state.

    Read-only eviction avoids buffer-sized dirty writebacks during the timed call.
    """
    # Imported on first use: Triton is needed only where a CUDA device is timed.
    from tileops._utils._l2_sweep import read_sweep

    reset_persisting_l2_cache()
    read_sweep(buffer)


def median_busy_ms(samples: "list[Sample]") -> float:
    """Median device-busy time shared by benchmark reporting and autotuning."""
    return statistics.median(s.device_busy_ms for s in samples)
