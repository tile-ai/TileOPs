"""How a benchmark calls each manifest op on its case.

Each family module holds ``ENTRIES`` for the ops whose case needs more than the default; they
merge here, and a name registered twice fails. The workload of an op's call comes from
``workloads.calls``.
"""

import dataclasses
import importlib
from typing import Any, Callable, Optional

from benchmarks.api import Implementation

__all__ = ["Entry", "default_binder", "entry"]


def default_binder(op: Any, case: Any) -> Implementation:
    """The op called on the case's inputs, which it must leave unchanged; it receives a
    ``kw_only`` input by name, as its ``forward`` takes it."""
    signature = case._call.signature

    def run(*inputs: Any) -> Any:
        args, keywords = signature.forward_arguments(inputs)
        return op(*args, **keywords)

    return Implementation(run=run)


@dataclasses.dataclass(frozen=True)
class Entry:
    """How a benchmark calls one op on its case.

    ``inputs`` is for an op whose inputs are not the workload's ``gen_inputs()``. ``binder``
    turns the op into the implementation recorded as ``tileops``.
    """

    count_copies: bool = False
    binder: Callable[[Any, Any], Implementation] = default_binder
    inputs: Optional[Callable[[Any], tuple]] = None


_FAMILIES = (
    "attention",
    "convolution",
    "elementwise",
    "gemm",
    "linear_attention",
    "mamba",
    "moe",
    "norm",
    "pool",
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
    """How a benchmark calls manifest op *name*: its registry entry, else the default."""
    return _load().get(name, Entry())
