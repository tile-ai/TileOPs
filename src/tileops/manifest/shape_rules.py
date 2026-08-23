"""Shared shape-rule helpers for reduction-style ops.

Concrete `shape_rules` expressions for the reduction family follow a
recurring pattern: validate the requested reduction dimension against
the input rank, normalize negatives via modulo, and (for sequence-valued
``dim``) enforce uniqueness of the normalized indices. Pasting Python
expressions into every manifest entry duplicates the contract and lets
the wording drift between ops.

This module exposes those patterns as named, individually testable
predicates and value extractors. Each is registered by name into the
validator's shape_rule builtin set (``scripts/validate_manifest.py``)
alongside the broadcasting helpers (``broadcast_shapes`` etc.) and the
Python primitives (``len``, ``range``, …), so a manifest entry can call
them by bare name::

    shape_rules:
      - "dim_range_validity(x, dim)"
      - "dim_uniqueness(x, dim)"
      - "output.ndim == x.ndim - len(reduced_axes(x, dim))"

Inline-string rules continue to work unchanged so ops can migrate one
at a time.

A malformed ``dim`` — say a list whose elements are not ints — propagates
its own ``TypeError``. The validator classifies such eval errors as
warnings: the rule counts as un-evaluatable under mock inputs and the
parity check is skipped.
"""

from __future__ import annotations

from typing import Any


def dim_range_validity(x: Any, dim: Any) -> bool:
    """Return True iff every requested axis lies within ``[-x.ndim, x.ndim)``.

    Mirrors the inline reduction-dim expression
    ``dim is None or all(-x.ndim <= d < x.ndim for d in
    ([dim] if isinstance(dim, int) else dim))`` exactly. ``dim is None``
    short-circuits to True (callers fall back to "all axes"); a single
    int wraps into a one-element list before the bounds check; any other
    value is iterated directly. Non-iterable values, or sequences whose
    elements cannot be compared to ``int``, propagate their own exception;
    the validator surfaces that as a warning (parity check skipped).

    Args:
        x: A tensor-like object exposing ``.ndim``.
        dim: An int, ``None``, or an iterable of ints.

    Returns:
        True when every requested axis is in range, False otherwise.

    Raises:
        TypeError: For malformed ``dim`` values (e.g. a list whose
            elements are strings); the validator handles this as a
            warning, matching the equivalent inline expression.
    """
    if dim is None:
        return True
    iterable = [dim] if isinstance(dim, int) else dim
    return all(-x.ndim <= d < x.ndim for d in iterable)


def dim_uniqueness(x: Any, dim: Any) -> bool:
    """Return True iff the normalized indices in ``dim`` are unique.

    Mirrors the inline reduction-dim expression
    ``isinstance(dim, (int, type(None))) or
    len({d % x.ndim for d in dim}) == len(dim)`` exactly. ``None`` and
    ``int`` short-circuit to True (a single axis cannot duplicate);
    sequence inputs normalize each entry via ``d % x.ndim`` and require
    the resulting set to retain the original cardinality. Non-iterable
    values, or sequences whose elements cannot be reduced modulo an int,
    propagate their own exception; the validator surfaces that as a
    warning (parity check skipped).

    Args:
        x: A tensor-like object exposing ``.ndim``.
        dim: An int, ``None``, or a sequence of ints. An empty sequence
            is trivially unique.

    Returns:
        True when no two requested axes collapse to the same normalized
        index, False otherwise.

    Raises:
        TypeError: For malformed ``dim`` values (e.g. a list of strings);
            the validator handles this as a warning, matching the pre-
            migration inline behaviour.
    """
    if isinstance(dim, (int, type(None))):
        return True
    return len({d % x.ndim for d in dim}) == len(dim)


def reduced_axes(x: Any, dim: Any) -> frozenset:
    """Return the set of normalized axis indices reduced over.

    Mirrors the structure of the inline reduction-axes expression
    ``{dim % x.ndim} if isinstance(dim, int) else
    {d % x.ndim for d in dim} if isinstance(dim, (list, tuple)) and
    len(dim) > 0 else set(range(x.ndim))``. The branch selection and
    per-element computation are identical; the only intentional change
    is the return type — every branch returns ``frozenset`` rather than
    ``set`` for immutability. Python treats ``frozenset == set`` of the
    same elements as equal, so manifest expressions that compare via
    ``==`` or use the result inside ``len(...)`` / ``in`` see no
    behaviour change.

    ``isinstance(dim, int)`` accepts ``bool`` too, since bool subclasses
    int. A list/tuple with elements that cannot be reduced modulo an int
    propagates its own exception; the validator surfaces that as a warning
    (parity check skipped). Anything else (``None``, a set, a string) falls
    through to the "all axes" branch.

    Pair this helper with `dim_range_validity` and
    `dim_uniqueness` so range / uniqueness violations surface
    before the output-shape rules consult `reduced_axes`.

    Args:
        x: A tensor-like object exposing ``.ndim``.
        dim: An int, ``None``, or a list/tuple of ints.

    Returns:
        A ``frozenset`` of normalized axis indices. The typical case
        returns ``frozenset[int]`` with members in ``[0, x.ndim)``; the
        return type is intentionally left unparameterised because
        malformed list/tuple elements pass through ``d % x.ndim`` and
        whatever that operation produces (e.g. ``float`` for ``1.5``)
        appears verbatim in the result. The helper preserves inline
        semantics, so callers that supply non-int sequence elements get
        the same return as the equivalent inline expression.

    Raises:
        TypeError: For a list/tuple whose elements cannot be reduced
            modulo ``x.ndim``; the validator handles this as a warning,
            matching the equivalent inline expression.
    """
    if isinstance(dim, int):
        return frozenset({dim % x.ndim})
    if isinstance(dim, (list, tuple)) and len(dim) > 0:
        return frozenset(d % x.ndim for d in dim)
    return frozenset(range(x.ndim))


class _Ranked:
    """The ``.ndim`` view of a shape tuple, so `reduced_axes` can read it."""

    __slots__ = ("ndim",)

    def __init__(self, ndim: int) -> None:
        self.ndim = ndim


def reduced_shape(shape: Any, dim: Any, keepdim: bool = False, empty_dim: str = "full") -> tuple:
    """Return the shape left after reducing *shape* over *dim*.

    The op layer and the manifest call the same function, so an op's
    ``_infer_output_shapes`` and its ``shape_rules`` cannot disagree about which
    axes a reduction removes.

    Args:
        shape: The input shape, as a sequence of ints.
        dim: An int, ``None``, or a list/tuple of ints, read the way
            `reduced_axes` reads it.
        keepdim: Whether a reduced axis stays as a length-1 axis.
        empty_dim: What an empty list/tuple means, matching the op layer's
            ``_empty_dim_policy``: ``"full"`` reduces every axis, ``"noop"``
            reduces none. An op that rejects an empty ``dim`` never gets here.

    Returns:
        The output shape. Reducing every axis without *keepdim* gives ``()``.
    """
    if empty_dim == "noop" and isinstance(dim, (list, tuple)) and len(dim) == 0:
        return tuple(shape)
    axes = reduced_axes(_Ranked(len(shape)), dim)
    if keepdim:
        return tuple(1 if i in axes else d for i, d in enumerate(shape))
    return tuple(d for i, d in enumerate(shape) if i not in axes)


__all__ = [
    "dim_range_validity",
    "dim_uniqueness",
    "reduced_axes",
    "reduced_shape",
]
