"""How a reduction op reads its ``dim`` param.

Turning ``dim`` into the axes a call reduces is the op's contract: which forms an empty
``dim`` takes, and which values a given rank accepts, differ per op and are declared in
the manifest. What a kernel then does with those axes — the permute to rows and back —
lives in `tileops.kernels.reduction._primitives`.
"""

from __future__ import annotations

from typing import Literal, Union

__all__ = [
    "normalize_dim",
]

EmptyDimPolicy = Literal["reject", "full", "noop"]


def normalize_dim(
    dim: Union[int, list[int], None],
    ndim: int,
    *,
    empty_dim_policy: EmptyDimPolicy = "reject",
) -> list[int]:
    """Normalize and validate a dim specification.

    Args:
        dim: Single int, list of ints, or ``None`` (reduce all dims).
        ndim: Number of dimensions in the input tensor.
        empty_dim_policy: ``"reject"`` (default) raises on ``dim=[] / ()``;
            ``"full"`` returns ``list(range(ndim))``; ``"noop"`` returns
            ``[]``, signaling the caller to short-circuit and return the
            input unchanged (modulo manifest-declared output-dtype cast).
            Each op opts in explicitly because shared callers have
            different empty-dim contracts.

    Returns:
        Sorted list of non-negative dim indices (ascending). An empty
        list is returned only when ``empty_dim_policy="noop"`` and the
        caller passed ``dim=[]`` / ``dim=()``.

    Raises:
        IndexError: If any dim is out of range.
        ValueError: If duplicate dims are given, or if ``dim`` is an
            empty list / tuple and ``empty_dim_policy="reject"``.
    """
    if dim is None:
        return list(range(ndim))

    dims = [dim] if isinstance(dim, int) else list(dim)

    if len(dims) == 0:
        if empty_dim_policy == "full":
            return list(range(ndim))
        if empty_dim_policy == "noop":
            # Caller MUST detect [] and short-circuit before entering kernel
            # paths -- the kernel does not handle a zero-dim reduction.
            return []
        raise ValueError(
            "dim=[] is not supported by this op; pass "
            'empty_dim_policy="full" to opt in to full-reduction '
            'or empty_dim_policy="noop" to opt in to the identity '
            "(return-input) contract."
        )

    normalized = []
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {d})"
            )
        normalized.append(d % ndim)

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate dims in reduction: {dims}")

    return sorted(normalized)
