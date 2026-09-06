"""The op body an elementwise kernel runs: how it is wrapped, and what it is named.

A ``@tilelang.jit`` builder may close over scalars only, so the body lives here and the
builder closes over its name.
"""

from typing import Callable

import tilelang.language as T

from ._dtype import BOOL_STORAGE_DTYPE

__all__ = ["op_func_for", "register_op_func"]

_OP_FUNCS: dict[str, Callable] = {}


def register_op_func(name: str, op_func: Callable) -> str:
    """Bind *op_func* to *name* and return the name.

    The name is the autotuner's cache key, so it must spell out everything the body
    depends on: class, dtypes, strategy, any baked-in constant.
    """
    _OP_FUNCS[name] = op_func
    return name


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


def _store_unary_bool_as_int8(op_func):
    return _store_bool_as_int8(op_func, arity=1)


def _store_binary_bool_as_int8(op_func):
    return _store_bool_as_int8(op_func, arity=2)
