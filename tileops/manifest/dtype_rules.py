"""Grammar for manifest ``signature`` dtype expressions.

A dtype expression is a concrete dtype name, a ``|`` union of tokens,
``same_as(<name>)``, or ``promote_int_to_float(<name>)``. Three consumers read
it and each used to carry its own regex; the copies had already drifted.

Accessors return the referenced *name*, not a dtype — this package depends on
nothing beyond the standard library and PyYAML.
"""

from __future__ import annotations

import re

#: Group 1 is the referenced tensor name.
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


__all__ = [
    "PROMOTE_INT_TO_FLOAT_RE",
    "SAME_AS_RE",
    "parse_tokens",
    "promote_int_to_float_ref",
    "same_as_ref",
]
