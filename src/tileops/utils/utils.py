import functools

import torch

# Lanes in a CUDA warp. Identical on every supported architecture (SM80-SM90)
# and baked into TIR at build time (loop bounds, shuffle widths), so it is a
# constant rather than a per-device query like the properties below.
WARP_LANES: int = 32
# Butterfly shuffle steps that reduce one warp: log2(WARP_LANES).
WARP_SHUFFLE_STAGES: int = WARP_LANES.bit_length() - 1

str2dtype = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
}


# Device properties are cached per device index, never process-wide: one process
# may drive several cards, and they need not be the same model or architecture.
# Both probes sit on the per-call kernel-selection path, so an uncached
# `get_device_name` string scan would run on every forward.


# Spin cycles queued before a device_busy_of measurement: tens of milliseconds
# on any supported clock, ample to enqueue every timed call first.
_BUSY_TIMING_SPIN_CYCLES = 50_000_000


# Calibrated boards: the key selection tables use -> the name fragment CUDA reports.
# All SKUs of a board share its key. GPU profiles match the full name instead
# (:func:`tileops.perf.find_profile`): a speed-of-light reading is not shared.
_CALIBRATION_BOARDS = {"h200": "H200"}


@functools.lru_cache(maxsize=16)
def _device_name(index: int) -> str:
    return torch.cuda.get_device_name(index)


@functools.lru_cache(maxsize=16)
def _sm_version(index: int) -> int:
    major, minor = torch.cuda.get_device_capability(index)
    return major * 10 + minor


def calibration_key(device_name: str) -> "str | None":
    """The key of the calibrated board *device_name* belongs to, or ``None``.

    The one place a device name is matched; fitted tuning data is keyed by the result.
    Case-insensitive.
    """
    upper = device_name.upper()
    return next((key for key, board in _CALIBRATION_BOARDS.items() if board in upper), None)


def device_calibration(index: "int | None" = None) -> "str | None":
    """:func:`calibration_key` of the device; defaults to the current device."""
    if not torch.cuda.is_available():
        return None
    return calibration_key(_device_name(torch.cuda.current_device() if index is None else index))


def get_sm_version(index: "int | None" = None) -> int:
    """Architecture of the device as ``major * 10 + minor``; defaults to current."""
    return _sm_version(torch.cuda.current_device() if index is None else index)


def get_sm_count(index: "int | None" = None) -> int:
    """Streaming-multiprocessor count of the device; defaults to current.

    Uncached: torch already caches device properties in C++, and the only
    callers read this once per kernel construction. Raises without a device,
    like :func:`get_sm_version`: kernel selection reads this number, and a
    stand-in for it decides a dispatch the caller cannot tell from a measured
    one.
    """
    device = torch.cuda.current_device() if index is None else index
    return torch.cuda.get_device_properties(device).multi_processor_count


def get_shared_memory_optin(index: "int | None" = None) -> int:
    """Shared memory one block of the device may take after opting in; defaults to current.

    Uncached and raising without a device, like :func:`get_sm_count`.
    """
    device = torch.cuda.current_device() if index is None else index
    return torch.cuda.get_device_properties(device).shared_memory_per_block_optin


@functools.lru_cache(maxsize=16)
def _device_facts(index: int) -> "tuple[int, str | None, int, int]":
    props = torch.cuda.get_device_properties(index)
    return (
        _sm_version(index),
        calibration_key(_device_name(index)),
        props.multi_processor_count,
        props.shared_memory_per_block_optin,
    )


def device_facts(index: "int | None" = None) -> "tuple[int, str | None, int, int]":
    """``(arch, calibration, sm_count, smem_budget)`` of the device; defaults to the current
    device. ``smem_budget`` is the shared memory one block may take after opting in.

    One cached lookup for a call record, which is built on the per-call path. The
    index is resolved before the cache is read, so ``None`` never names whichever
    device happened to be current on the first call.
    """
    return _device_facts(torch.cuda.current_device() if index is None else index)


def forget_device_properties() -> None:
    """Drop the cached architecture and name of every device.

    A device's properties do not change, so the cache is normally never
    invalidated. What does change is whether the query itself can be answered —
    a caller that must observe a failing probe has to reach it before the first
    successful one is remembered.
    """
    _device_name.cache_clear()
    _sm_version.cache_clear()
    _device_facts.cache_clear()


def device_busy_of(call, device: "torch.device", warmup: int = 5, rep: int = 20) -> float:
    """Mean device time of *call* in milliseconds with host gaps excluded.

    Judges paths that launch different kernel counts by their GPU work alone;
    wall latency would charge a multi-launch path the host gaps between its
    launches. A spin kernel holds the device while every timed call is
    enqueued, so the queue then drains back to back and the event pair brackets
    execution only. Deliberately not a profiler: the benchmark's own collector
    owns the process's CUPTI subscription, and a second subscriber would break
    its kernel attribution for the rest of the process.
    """
    with torch.cuda.device(device):
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(_BUSY_TIMING_SPIN_CYCLES)
        start.record()
        for _ in range(rep):
            call()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / rep
