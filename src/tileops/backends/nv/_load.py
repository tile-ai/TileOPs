"""Turning a binding string into a kernel class, no earlier than necessary."""

from __future__ import annotations

import functools
import importlib

from tileops.backend import Kernel

from ._bindings import BINDINGS


@functools.lru_cache(maxsize=None)
def kernel_class(op: str, role: str) -> type[Kernel]:
    """Import and return the class bound to *role* of *op*.

    First call imports the kernel module, and TileLang with it. Cached, so an op that
    builds several specializations pays the import once.

    Raises:
        KeyError: This backend has no binding for that op or role.
    """
    module, _, name = BINDINGS[op][role].partition(":")
    return getattr(importlib.import_module(module), name)
