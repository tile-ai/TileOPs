"""The op body an elementwise kernel runs: how it is wrapped, and what it is named.

A ``@tilelang.jit`` builder may close over scalars only, so the body lives here and the
builder closes over its name.
"""

from dataclasses import dataclass
from typing import Callable

import tilelang.language as T
import tvm.tirx as tirx

from tileops._kernels.elementwise._dtype import BOOL_STORAGE_DTYPE

__all__ = ["GuardedOpFunc", "op_func_for", "register_op_func", "single_instruction_ops"]

_OP_FUNCS: dict[str, Callable] = {}


def register_op_func(name: str, op_func: Callable) -> str:
    """Bind *op_func* to *name* and return the name.

    The name is the autotuner's cache key, so it must spell out everything the body
    depends on: class, dtypes, strategy, any baked-in constant.
    """
    _OP_FUNCS[name] = op_func
    return name


@dataclass(frozen=True)
class GuardedOpFunc:
    """An op body with a cheaper form that equals it wherever a guard holds.

    ``fast(a, b)`` returns ``(value, holds)``. A staged builder runs ``fast`` over a
    thread's elements and calls the body only in a thread where ``holds`` fails for
    one of them, which takes the body's own test and fallback off the common path.
    """

    body: Callable
    fast: Callable

    def __call__(self, *args):
        return self.body(*args)


def op_func_for(name: str) -> Callable:
    """The op body registered under *name*."""
    return _OP_FUNCS[name]


def _store_bool_as_int8(op_func, arity: int):
    if arity == 1:

        def wrapped(x):
            return T.if_then_else(
                op_func(x),
                T.cast(1, BOOL_STORAGE_DTYPE),
                T.cast(0, BOOL_STORAGE_DTYPE),
            )
    else:

        def wrapped(a, b):
            return T.if_then_else(
                op_func(a, b),
                T.cast(1, BOOL_STORAGE_DTYPE),
                T.cast(0, BOOL_STORAGE_DTYPE),
            )

    return wrapped


def store_unary_bool_as_int8(op_func):
    return _store_bool_as_int8(op_func, arity=1)


def store_binary_bool_as_int8(op_func):
    return _store_bool_as_int8(op_func, arity=2)


# Nodes that lower to one instruction an element: add, subtract, multiply, min, max,
# compares and logic; and calls that do: a select, a bitwise operation, an absolute value.
_SINGLE_INSTRUCTION_NODES = (
    tirx.Add,
    tirx.Sub,
    tirx.Mul,
    tirx.Min,
    tirx.Max,
    tirx.EQ,
    tirx.NE,
    tirx.LT,
    tirx.LE,
    tirx.GT,
    tirx.GE,
    tirx.And,
    tirx.Or,
    tirx.Not,
    tirx.Select,
)
_SINGLE_INSTRUCTION_CALLS = frozenset(
    {
        "tirx.if_then_else",
        "tirx.bitwise_and",
        "tirx.bitwise_or",
        "tirx.bitwise_xor",
        "tirx.bitwise_not",
        "tirx.fabs",
    }
)


def single_instruction_ops(op_func: Callable, dtype: str, arity: int) -> "int | None":
    """An estimate of the instructions *op_func* issues an element in *dtype*: the count of its
    operations when each is one of the single-instruction kinds above, else ``None``.

    ``None`` also stands for every body this cannot read as one expression: a guarded body,
    whose fast form, guard and fallback the builder runs in turn, and a body that does not
    evaluate on bare variables. Those keep the default launch, and their builder reports
    any error of their own.
    """
    if isinstance(op_func, GuardedOpFunc):
        return None
    try:
        expr = op_func(*(tirx.Var(f"operand{i}", dtype) for i in range(arity)))
    except Exception:
        return None
    if not isinstance(expr, tirx.PrimExpr):
        return None
    count, single = [0], [True]

    def visit(node):
        if isinstance(node, (tirx.Var, tirx.IntImm, tirx.FloatImm, tirx.StringImm)):
            return
        if isinstance(node, tirx.Call):
            name = getattr(node.op, "name", None)
            single[0] = single[0] and str(name) in _SINGLE_INSTRUCTION_CALLS
        elif not isinstance(node, _SINGLE_INSTRUCTION_NODES):
            single[0] = False
        count[0] += 1

    tirx.stmt_functor.post_order_visit(expr, visit)
    return count[0] if single[0] else None
