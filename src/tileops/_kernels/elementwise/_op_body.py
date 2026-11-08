"""The op body an elementwise kernel runs: how it is wrapped, and what it is named.

A ``@tilelang.jit`` builder may close over scalars only, so the body lives here and the
builder closes over its name.
"""

from dataclasses import dataclass
from typing import Callable

import tilelang.language as T
import tvm.tirx as tirx

from tileops._kernels.elementwise._dtype import BOOL_STORAGE_DTYPE

__all__ = [
    "GuardedOpFunc",
    "LaneMaskedOpFunc",
    "body_operations",
    "is_simple_unary",
    "op_func_for",
    "register_op_func",
]

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


@dataclass(frozen=True)
class LaneMaskedOpFunc:
    """A bool-result op body with a form over two 16-bit lanes of a uint32 word.

    ``lanes(a_word, b_word)`` returns 0xffff in each lane where the body holds. A
    builder storing a thread's run of bools builds each 4-byte word from two such
    masks instead of one select per byte.
    """

    body: Callable
    lanes: Callable

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


_SIMPLE_UNARY_NODES = (
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
_SIMPLE_UNARY_CALLS = frozenset(
    {
        "tirx.if_then_else",
        "tirx.bitwise_and",
        "tirx.bitwise_or",
        "tirx.bitwise_xor",
        "tirx.bitwise_not",
        "tirx.fabs",
    }
)


def body_operations(op_func: Callable, *dtypes: str) -> "list | None":
    """The TIR operations *op_func* builds on one operand of each of *dtypes*, leaves left out.

    ``None`` where the body does not trace to one expression.
    """
    try:
        expr = op_func(*(tirx.Var(f"operand{i}", dt) for i, dt in enumerate(dtypes)))
    except Exception:
        return None
    if not isinstance(expr, tirx.PrimExpr):
        return None
    nodes = []
    tirx.stmt_functor.post_order_visit(expr, nodes.append)
    leaves = (tirx.Var, tirx.IntImm, tirx.FloatImm, tirx.StringImm)
    return [node for node in nodes if not isinstance(node, leaves)]


def is_simple_unary(op_func: Callable, dtype: str) -> bool:
    """Whether a unary expression has at most two inexpensive TIR operations."""
    if isinstance(op_func, GuardedOpFunc):
        return False
    operations = body_operations(op_func, dtype)
    if operations is None:
        return False
    return len(operations) <= 2 and all(
        str(getattr(node.op, "name", None)) in _SIMPLE_UNARY_CALLS
        if isinstance(node, tirx.Call)
        else isinstance(node, _SIMPLE_UNARY_NODES)
        for node in operations
    )
