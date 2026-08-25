"""Names for the non-scalar values an elementwise kernel body reads.

A ``@tilelang.jit`` builder may close over scalars only, so an op body or a stride
tuple lives here and the builder closes over its name.
"""

from dataclasses import dataclass
from typing import Callable

__all__ = [
    "BroadcastPlan",
    "broadcast_plan_for",
    "op_func_for",
    "register_broadcast_plan",
    "register_op_func",
]

_OP_FUNCS: dict[str, Callable] = {}
_BROADCAST_PLANS: dict[str, "BroadcastPlan"] = {}


@dataclass(frozen=True)
class BroadcastPlan:
    """The coalesced shape and per-operand strides a binary kernel indexes with."""

    coalesced_shape: tuple[int, ...]
    a_strides: tuple[int, ...]
    b_strides: tuple[int, ...]


def register_op_func(name: str, op_func: Callable) -> str:
    """Bind *op_func* to *name* and return the name.

    The name must spell out everything the body depends on -- class, dtypes, strategy,
    any baked-in constant -- because it is the autotuner's cache key: two bodies under
    one name would read as one kernel that has already been tuned.
    """
    _OP_FUNCS[name] = op_func
    return name


def op_func_for(name: str) -> Callable:
    """The op body registered under *name*."""
    return _OP_FUNCS[name]


def register_broadcast_plan(plan: BroadcastPlan) -> str:
    """Bind *plan* to a name derived from its own contents, and return that name."""
    name = f"{plan.coalesced_shape}|{plan.a_strides}|{plan.b_strides}"
    _BROADCAST_PLANS[name] = plan
    return name


def broadcast_plan_for(name: str) -> BroadcastPlan:
    """The plan registered under *name*."""
    return _BROADCAST_PLANS[name]
