"""NPU device abstraction layer.

Wraps torch_npu so the rest of the project uses a uniform API without
sprinkling ``torch.npu`` calls everywhere.  Falls back to CUDA when
``NPU_BENCHMARK_FORCE_CUDA=1`` is set (for local dev on NVIDIA GPUs).

Requires ``torch_npu`` to be installed for Ascend NPU support.
"""

from __future__ import annotations

import os

import torch

_FORCE_CUDA = os.getenv("NPU_BENCHMARK_FORCE_CUDA", "0") == "1"

_HAS_NPU = hasattr(torch, "npu")
if _HAS_NPU:
    try:
        _NPU_AVAILABLE = torch.npu.is_available()
    except Exception:
        _NPU_AVAILABLE = False
else:
    _NPU_AVAILABLE = False

if _FORCE_CUDA and torch.cuda.is_available():
    _BACKEND = "cuda"
elif _HAS_NPU and _NPU_AVAILABLE:
    _BACKEND = "npu"
elif torch.cuda.is_available():
    _BACKEND = "cuda"
else:
    _BACKEND = "cpu"


def backend() -> str:
    return _BACKEND


def is_available() -> bool:
    return _BACKEND in ("npu", "cuda")


def device() -> torch.device:
    return torch.device(_BACKEND)


def device_str() -> str:
    return _BACKEND


def synchronize() -> None:
    if _BACKEND == "npu":
        torch.npu.synchronize()
    elif _BACKEND == "cuda":
        torch.cuda.synchronize()


def empty_cache() -> None:
    if _BACKEND == "npu":
        torch.npu.empty_cache()
    elif _BACKEND == "cuda":
        torch.cuda.empty_cache()


def manual_seed_all(seed: int) -> None:
    if _BACKEND == "npu":
        torch.npu.manual_seed_all(seed)
    elif _BACKEND == "cuda":
        torch.cuda.manual_seed_all(seed)


def get_device_name(idx: int = 0) -> str:
    if _BACKEND == "npu":
        return torch.npu.get_device_name(idx)
    elif _BACKEND == "cuda":
        return torch.cuda.get_device_name(idx)
    return "cpu"


def get_device_properties(idx: int = 0):
    if _BACKEND == "npu":
        return torch.npu.get_device_properties(idx)
    elif _BACKEND == "cuda":
        return torch.cuda.get_device_properties(idx)
    raise RuntimeError("no accelerator device available")


def timing_event(enable_timing: bool = True):
    """Create a timing event on the active backend."""
    if _BACKEND == "npu":
        return torch.npu.Event(enable_timing=enable_timing)
    elif _BACKEND == "cuda":
        return torch.cuda.Event(enable_timing=enable_timing)
    raise RuntimeError("timing events require an accelerator device")


def is_npu() -> bool:
    return _BACKEND == "npu"


def is_cuda() -> bool:
    return _BACKEND == "cuda"
