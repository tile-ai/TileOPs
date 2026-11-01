"""The workload of each manifest op's call: the one map benchmarks and test targets read.

Each family module holds ``WORKLOADS``, a factory per op taking the op's manifest ``Call``;
they merge here, and an op registered twice fails.
"""

import functools
import importlib
from typing import Any, Callable

__all__ = ["call_workload", "registered"]

_FAMILIES = (
    "attention",
    "convolution",
    "elementwise",
    "engram",
    "fft",
    "gemm",
    "linear_attention",
    "mamba",
    "mhc",
    "moe",
    "norm",
    "pool",
    "quantization",
    "reduction",
    "rope",
    "sampling",
    "transform",
)


@functools.cache
def _load(families: tuple[str, ...] = _FAMILIES) -> dict[str, Callable[[Any], Any]]:
    merged: dict[str, Callable[[Any], Any]] = {}
    for family in families:
        for name, factory in importlib.import_module(f"workloads.calls.{family}").WORKLOADS.items():
            if name in merged:
                raise ValueError(f"{name} is registered twice; the second is in {family}")
            merged[name] = factory
    return merged


def call_workload(name: str) -> Callable[[Any], Any]:
    """The factory building the workload of a manifest call of op *name*."""
    try:
        return _load()[name]
    except KeyError:
        raise KeyError(f"{name} has no call workload; register it in workloads/calls/") from None


def registered() -> frozenset[str]:
    """Every op with a call workload."""
    return frozenset(_load())
