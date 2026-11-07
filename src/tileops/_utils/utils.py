import functools
from typing import NamedTuple

import torch

# Lanes in a CUDA warp. Identical on every supported architecture (SM80-SM90)
# and baked into TIR at build time (loop bounds, shuffle widths), so it is a
# constant rather than a per-device query like the properties below.
WARP_LANES: int = 32
# Butterfly shuffle steps that reduce one warp: log2(WARP_LANES).
WARP_SHUFFLE_STAGES: int = WARP_LANES.bit_length() - 1
# CTAs one SM holds at once, by architecture (CUDA C++ Programming Guide, compute capabilities).
RESIDENT_CTAS: dict[int, int] = {80: 32, 86: 16, 89: 24, 90: 32, 100: 32}

STR_TO_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
}


# Calibrated boards: the key selection tables use -> the name fragment CUDA reports.
# All SKUs of a board share its key. GPU profiles match the full name instead
# (:func:`tileops._perf.find_profile`): a speed-of-light reading is not shared.
_CALIBRATION_BOARDS = {"h200": "H200"}


def calibration_key(device_name: str) -> "str | None":
    """The key of the calibrated board *device_name* belongs to, or ``None``.

    The one place a device name is matched; fitted tuning data is keyed by the result.
    Case-insensitive.
    """
    upper = device_name.upper()
    return next((key for key, board in _CALIBRATION_BOARDS.items() if board in upper), None)


class CudaFacts(NamedTuple):
    """What kernel selection and builds read about one CUDA device."""

    # ``major * 10 + minor``.
    arch: int
    # :func:`calibration_key` of the device's name.
    calibration: "str | None"
    sm_count: int
    # Shared memory one block may take after opting in, in bytes.
    smem_budget: int
    threads_per_sm: int
    regs_per_sm: int
    # Shared memory of one SM, in bytes.
    smem_per_sm: int
    l2_bytes: int


# Cached per device index, never process-wide: one process may drive several cards, and
# they need not be the same model or architecture. Selection reads these on the per-call
# path, so an uncached `get_device_name` string scan would run on every forward.
@functools.lru_cache(maxsize=16)
def _device_facts(index: int) -> CudaFacts:
    major, minor = torch.cuda.get_device_capability(index)
    props = torch.cuda.get_device_properties(index)
    return CudaFacts(
        arch=major * 10 + minor,
        calibration=calibration_key(torch.cuda.get_device_name(index)),
        sm_count=props.multi_processor_count,
        smem_budget=props.shared_memory_per_block_optin,
        threads_per_sm=props.max_threads_per_multi_processor,
        regs_per_sm=props.regs_per_multiprocessor,
        smem_per_sm=props.shared_memory_per_multiprocessor,
        l2_bytes=props.L2_cache_size,
    )


def device_facts(index: "int | None" = None) -> CudaFacts:
    """The facts of CUDA device *index*; defaults to the current device.

    The one reader of CUDA device facts: call specs, build targets and the helpers below
    all read through it. The index is resolved before the cache is read, so ``None`` never
    names whichever device happened to be current on the first call. Raises without a
    device: a stand-in for a fact decides a dispatch the caller cannot tell from a measured
    one.
    """
    return _device_facts(torch.cuda.current_device() if index is None else index)


def device_calibration(index: "int | None" = None) -> "str | None":
    """The ``calibration`` of the device, or ``None`` without CUDA; defaults to current."""
    if not torch.cuda.is_available():
        return None
    return device_facts(index).calibration


def get_sm_version(index: "int | None" = None) -> int:
    """The ``arch`` of the device; defaults to current."""
    return device_facts(index).arch


def get_sm_count(index: "int | None" = None) -> int:
    """The ``sm_count`` of the device; defaults to current."""
    return device_facts(index).sm_count


def get_shared_memory_optin(index: "int | None" = None) -> int:
    """The ``smem_budget`` of the device; defaults to current."""
    return device_facts(index).smem_budget


def forget_device_properties() -> None:
    """Drop the cached accelerator type and the facts of every device.

    A device's properties do not change, so the cache is normally never
    invalidated. What does change is whether the query itself can be answered —
    a caller that must observe a failing probe has to reach it before the first
    successful one is remembered.
    """
    from tileops._utils.device import current_accelerator_type

    _device_facts.cache_clear()
    current_accelerator_type.cache_clear()
