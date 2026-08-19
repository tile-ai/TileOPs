import functools

import torch

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


@functools.lru_cache(maxsize=16)
def _device_name(index: int) -> str:
    return torch.cuda.get_device_name(index).upper()


@functools.lru_cache(maxsize=16)
def _sm_version(index: int) -> int:
    major, minor = torch.cuda.get_device_capability(index)
    return major * 10 + minor


@functools.lru_cache(maxsize=16)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def is_h200(index: "int | None" = None) -> bool:
    """Whether the device is an H200; defaults to the current device."""
    if not torch.cuda.is_available():
        return False
    return "H200" in _device_name(torch.cuda.current_device() if index is None else index)


def get_sm_version(index: "int | None" = None) -> int:
    """Architecture of the device as ``major * 10 + minor``; defaults to current."""
    return _sm_version(torch.cuda.current_device() if index is None else index)


def get_sm_count(index: "int | None" = None) -> int:
    """Multiprocessors on the device; defaults to current.

    Persistent kernels size their grid by this, so it is read once per kernel
    construction rather than kept as a constant per kernel.
    """
    return _sm_count(torch.cuda.current_device() if index is None else index)


def forget_device_properties() -> None:
    """Drop the cached architecture, multiprocessor count and name of every device.

    A device's properties do not change, so the cache is normally never
    invalidated. What does change is whether the query itself can be answered —
    a caller that must observe a failing probe has to reach it before the first
    successful one is remembered.
    """
    _device_name.cache_clear()
    _sm_version.cache_clear()
    _sm_count.cache_clear()
