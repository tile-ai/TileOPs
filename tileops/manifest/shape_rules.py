"""Shared shape-rule helpers for reduction-style ops.

Concrete `shape_rules` expressions for the reduction family follow a
recurring pattern: validate the requested reduction dimension against the
input rank, normalize negatives via modulo, and (for sequence-valued
``dim``) enforce uniqueness of the normalized indices. Pasting Python
expressions into every manifest entry duplicates the contract and lets
the wording drift between ops.

This module exposes those patterns as named, individually testable
predicates. Manifest entries opt in via the ``helper:`` URI prefix; for
example::

    shape_rules:
      - "helper:dim_range_validity(x, dim)"
      - "helper:dim_uniqueness(x, dim)"

The validator resolves each ``helper:NAME(args)`` rule by stripping the
prefix and evaluating the call against the same context used for
inline-string rules. The helpers themselves are exposed to that context
via :data:`HELPERS`. Inline-string rules continue to work unchanged so
ops can migrate one at a time.

Each helper accepts the primitive forms its caller has on hand
(``tensor`` objects with a ``.ndim`` attribute, plus the raw ``dim``
value). The helpers never raise: an out-of-range or malformed ``dim`` is
reported as ``False`` so the validator's normal failure path surfaces it
as a shape-rule violation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _ndim_of(x: Any) -> int | None:
    """Return ``x.ndim`` if available, else ``None``.

    The validator's mock shape context exposes ``ndim`` on tensor stand-ins,
    matching torch's tensor surface. Returning ``None`` for malformed
    inputs lets callers surface the issue as a rule failure rather than a
    Python exception that would be classified as an eval skip.
    """
    try:
        ndim = int(x.ndim)
    except (AttributeError, TypeError, ValueError):
        return None
    return ndim


def _iter_dims(dim: Any) -> tuple[int, ...] | None:
    """Coerce ``dim`` to a tuple of ints.

    Returns:
        - ``None`` when ``dim is None`` (caller decides how to interpret).
        - A single-element tuple when ``dim`` is an int.
        - A tuple of ints for list / tuple inputs whose every element is
          an int.
        - ``None`` for any other shape, signalling a malformed value.
    """
    if dim is None:
        return None
    if isinstance(dim, bool):  # bool is a subclass of int; reject explicitly
        return None
    if isinstance(dim, int):
        return (dim,)
    if isinstance(dim, Sequence):
        out: list[int] = []
        for d in dim:
            if isinstance(d, bool) or not isinstance(d, int):
                return None
            out.append(d)
        return tuple(out)
    return None


def dim_range_validity(x: Any, dim: Any) -> bool:
    """Return True iff ``dim`` lies within ``[-x.ndim, x.ndim)``.

    Implements the reduction-family contract: ``dim is None`` is always
    valid (callers fall back to a default axis); a single int must satisfy
    ``-x.ndim <= dim < x.ndim``; a sequence must have every element
    satisfy the same bound. An empty sequence is treated as valid by this
    predicate (use :func:`dim_uniqueness` to enforce non-emptiness when
    the op requires it; PyTorch's reduction semantics for an empty ``dim``
    list reduce over all axes).

    Args:
        x: A tensor-like object exposing ``.ndim``.
        dim: An int, ``None``, or a sequence of ints.

    Returns:
        True when every requested axis is in range. False on any
        out-of-range index, on a malformed ``dim`` value, or when
        ``x.ndim`` cannot be read.
    """
    ndim = _ndim_of(x)
    if ndim is None:
        return False
    if dim is None:
        return True
    dims = _iter_dims(dim)
    if dims is None:
        return False
    return all(-ndim <= d < ndim for d in dims)


def dim_uniqueness(x: Any, dim: Any) -> bool:
    """Return True iff the normalized indices in ``dim`` are unique.

    For sequence-valued ``dim``, every entry is normalized via ``d %
    x.ndim`` and the resulting set must have the same length as the
    original sequence (no duplicates after sign normalization). A single
    int and ``None`` always pass (one axis, or "all axes" — neither can
    duplicate).

    The predicate is independent of :func:`dim_range_validity`; callers
    typically pair the two so range failures are reported separately
    from uniqueness failures.

    Args:
        x: A tensor-like object exposing ``.ndim``.
        dim: An int, ``None``, or a sequence of ints. A sequence may be
            empty; an empty sequence is trivially unique.

    Returns:
        True when no two requested axes collapse to the same normalized
        index. False on duplicates, on a malformed ``dim`` value, or
        when ``x.ndim`` cannot be read.
    """
    ndim = _ndim_of(x)
    if ndim is None:
        return False
    if dim is None or isinstance(dim, int) and not isinstance(dim, bool):
        return True
    dims = _iter_dims(dim)
    if dims is None:
        return False
    if not dims:
        return True
    normalized = {d % ndim for d in dims}
    return len(normalized) == len(dims)


HELPERS: dict[str, Any] = {
    "dim_range_validity": dim_range_validity,
    "dim_uniqueness": dim_uniqueness,
}
"""Registry of helper names exposed to the manifest's ``helper:`` URI scheme.

The validator merges this mapping into the ``shape_rules`` evaluation
context so a rule like ``helper:dim_range_validity(x, dim)`` resolves
the call without any extra plumbing per op.
"""


__all__ = [
    "HELPERS",
    "dim_range_validity",
    "dim_uniqueness",
]
