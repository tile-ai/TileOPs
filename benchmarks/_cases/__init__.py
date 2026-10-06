"""How each manifest op becomes a benchmark case: one registry entry per op.

An entry names the workload a call builds, whether the case counts device copies, and how
the TileOps op is bound into an :class:`~benchmarks.api.Implementation`. Each family module
holds ``ENTRIES``; they merge here, and a name registered twice fails on import.
"""

import dataclasses
import importlib
from typing import Any, Callable, Optional

from benchmarks.api import Implementation

__all__ = ["Entry", "entry"]


def default_binder(op: Any, case: Any) -> Implementation:
    """The op called on the case's inputs, which it must leave unchanged."""
    return Implementation(run=op)


@dataclasses.dataclass(frozen=True)
class Entry:
    """How one op's manifest call becomes a case.

    ``workload`` builds the call's workload. ``inputs`` is for an op whose inputs are not the
    workload's ``gen_inputs()``. ``binder`` turns the op into the implementation recorded as
    ``tileops``.
    """

    workload: Callable[[Any], Any]
    count_copies: bool = False
    binder: Callable[[Any, Any], Implementation] = default_binder
    inputs: Optional[Callable[[Any], tuple]] = None


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
)


@dataclasses.dataclass
class _Registry:
    entries: Optional[dict[str, Entry]] = None


_registry = _Registry()


def _load() -> dict[str, Entry]:
    if _registry.entries is None:
        merged: dict[str, Entry] = {}
        for family in _FAMILIES:
            for name, value in importlib.import_module(
                f"benchmarks._cases.{family}"
            ).ENTRIES.items():
                if name in merged:
                    raise ValueError(f"{name} is registered twice; the second is in {family}")
                merged[name] = value
        _registry.entries = merged
    return _registry.entries


def entry(name: str) -> Entry:
    """The registry entry of manifest op *name*."""
    try:
        return _load()[name]
    except KeyError:
        raise KeyError(
            f"{name} has no benchmark case factory; register it in benchmarks/_cases/"
        ) from None


def registered() -> frozenset[str]:
    """Every op with an entry."""
    return frozenset(_load())
