"""Shared parsers for manifest ``signature`` dtype expressions.

A dtype expression is one of:

- a concrete dtype name — ``"float16"``;
- a ``|`` union of tokens — ``"float16 | bfloat16 | same_as(input)"``;
- ``same_as(<name>)`` — follow another tensor in the same signature;
- ``promote_int_to_float(<name>)`` — follow another tensor, promoting
  integral dtypes to float, matching PyTorch's int-input promotion.

Several consumers read this grammar, and each used to carry its own copy of
the regex. The copies had already drifted apart. This module owns the
grammar so the wording cannot diverge again.

The parsers return the *referenced name* rather than a dtype: this package
depends on nothing beyond the standard library and PyYAML, so mapping a name
onto a concrete dtype stays with the caller.
"""

from __future__ import annotations

import re

_SAME_AS_RE = re.compile(r"^same_as\(\s*(\w+)\s*\)$")
_PROMOTE_INT_TO_FLOAT_RE = re.compile(r"^promote_int_to_float\(\s*(\w+)\s*\)$")


def parse_tokens(expr: str) -> list[str]:
    """Split a dtype expression into its ``|``-separated tokens.

    Args:
        expr: A dtype expression, e.g. ``"float16 | same_as(input)"``.

    Returns:
        The non-empty tokens, stripped of surrounding whitespace.
    """
    return [t.strip() for t in expr.split("|") if t.strip()]


def same_as_ref(token: str) -> str | None:
    """Return the tensor named by a ``same_as(...)`` token.

    Args:
        token: A single dtype token, already split out of any union.

    Returns:
        The referenced tensor name, or ``None`` if *token* is not a
        ``same_as`` expression.
    """
    m = _SAME_AS_RE.match(token)
    return m.group(1) if m else None


def promote_int_to_float_ref(token: str) -> str | None:
    """Return the tensor named by a ``promote_int_to_float(...)`` token.

    Args:
        token: A single dtype token, already split out of any union.

    Returns:
        The referenced tensor name, or ``None`` if *token* is not a
        ``promote_int_to_float`` expression.
    """
    m = _PROMOTE_INT_TO_FLOAT_RE.match(token)
    return m.group(1) if m else None


__all__ = ["parse_tokens", "promote_int_to_float_ref", "same_as_ref"]
