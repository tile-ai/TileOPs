"""Grammar for manifest ``signature`` dtype expressions.

A dtype expression is a concrete dtype name, a ``|`` union of tokens,
``same_as(<name>)``, or ``promote_int_to_float(<name>)``. This module is the one
place that parses it; its three consumers read the accessors below.

Accessors return the referenced *name*, not a dtype — this package depends on
nothing beyond the standard library and PyYAML.
"""

from __future__ import annotations

import re

__all__ = [
    "DTYPE_BITS",
    "PROMOTE_INT_TO_FLOAT_RE",
    "SAME_AS_RE",
    "parse_tokens",
    "promote_int_to_float_ref",
    "same_as_ref",
]

# The dtype registry: every dtype name a manifest may write, with its bits per element.
DTYPE_BITS: dict[str, int] = {
    "bool": 8,
    "uint8": 8,
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
    "float16": 16,
    "bfloat16": 16,
    "float32": 32,
    "float64": 64,
    "complex64": 64,
    "complex128": 128,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float8_e4m3": 8,
    "float8_e5m2fnuz": 8,
    "float8_e4m3fnuz": 8,
}

# Group 1 is the referenced tensor name.
SAME_AS_RE = re.compile(r"^same_as\(\s*(\w+)\s*\)$")
PROMOTE_INT_TO_FLOAT_RE = re.compile(r"^promote_int_to_float\(\s*(\w+)\s*\)$")


def parse_tokens(expr: str) -> list[str]:
    """Split a dtype expression into its non-empty ``|``-separated tokens."""
    return [t.strip() for t in expr.split("|") if t.strip()]


def same_as_ref(token: str) -> str | None:
    """Return the name in a ``same_as(...)`` token, or None if not one."""
    m = SAME_AS_RE.match(token)
    return m.group(1) if m else None


def promote_int_to_float_ref(token: str) -> str | None:
    """Return the name in a ``promote_int_to_float(...)`` token, or None."""
    m = PROMOTE_INT_TO_FLOAT_RE.match(token)
    return m.group(1) if m else None
