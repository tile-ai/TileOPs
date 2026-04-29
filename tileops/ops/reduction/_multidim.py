"""Multi-dim reduction utilities for the Op layer.

Provides helpers to flatten multiple reduction dims into a single dim,
so that existing single-dim kernels can handle ``dim=list[int]`` by
reducing over the flattened dimension.

Strategy: move all target dims to the end, flatten them into one dim,
reduce along that single dim, then restore the output shape with or
without keepdim.
"""

from __future__ import annotations

from typing import Literal, Union

import torch

__all__ = [
    "flatten_for_multidim",
    "normalize_dim",
    "restore_multidim_shape",
]

EmptyDimPolicy = Literal["reject", "full"]


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
        empty_dim_policy: How to treat ``dim=[]`` / ``dim=()``.
            ``"reject"`` (default) raises ``ValueError`` — caller's
            PyTorch / manifest contract does not define empty-dim
            semantics. ``"full"`` returns ``list(range(ndim))``,
            matching PyTorch ops whose contract treats empty-dim as
            full reduction (e.g. ``torch.sum``, ``torch.mean``,
            ``torch.linalg.vector_norm``). The default is ``"reject"``
            because shared callers (``_SoftmaxBaseOp``, ``logsumexp``,
            logical ops) have *different* empty-dim contracts; each
            op opts in explicitly.

    Returns:
        Sorted list of non-negative dim indices (ascending).

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
        raise ValueError(
            "dim=[] is not supported by this op; the helper is "
            "policy-neutral. Pass empty_dim_policy=\"full\" if your op's "
            "PyTorch / manifest contract treats empty-dim as full reduction."
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


def flatten_for_multidim(
    x: torch.Tensor, dims: list[int],
) -> tuple[torch.Tensor, torch.Size, list[int]]:
    """Move target dims to end and flatten them into one dim.

    Args:
        x: Input tensor (any shape).
        dims: Sorted list of non-negative dim indices to reduce.

    Returns:
        (x_reshaped, orig_shape, kept_dims) where:
        - x_reshaped has the target dims flattened into the last dim
        - orig_shape is the original tensor shape
        - kept_dims is the list of dims NOT being reduced (in order)
    """
    orig_shape = x.shape
    ndim = x.ndim

    # Determine which dims are kept vs reduced.
    kept_dims = [i for i in range(ndim) if i not in dims]

    # Permute: kept dims first, then reduced dims.
    perm = kept_dims + dims
    x = x.permute(perm).contiguous()

    # Compute shapes.
    kept_shape = [orig_shape[i] for i in kept_dims]
    reduced_size = 1
    for d in dims:
        reduced_size *= orig_shape[d]

    # Flatten reduced dims into one.
    # All dims being reduced -> (1, reduced_size); otherwise append flattened dim.
    new_shape = kept_shape + [reduced_size] if kept_shape else [1, reduced_size]

    x = x.reshape(new_shape)
    return x, orig_shape, kept_dims


def restore_multidim_shape(
    y: torch.Tensor,
    orig_shape: torch.Size,
    dims: list[int],
    keepdim: bool,
) -> torch.Tensor:
    """Reshape the output of a single-dim reduction back to multi-dim output shape.

    Args:
        y: Output tensor from single-dim reduction (reduced dim removed).
        orig_shape: Original input tensor shape.
        dims: Sorted list of non-negative dim indices that were reduced.
        keepdim: Whether to retain reduced dims as size 1.

    Returns:
        Tensor with the correct output shape.
    """
    ndim = len(orig_shape)
    kept_dims = [i for i in range(ndim) if i not in dims]

    if keepdim:
        # Build shape with 1 at each reduced dim position.
        out_shape = list(orig_shape)
        for d in dims:
            out_shape[d] = 1
        return y.reshape(out_shape)
    else:
        # Output shape is just the kept dims.
        out_shape = [orig_shape[i] for i in kept_dims]
        if not out_shape:
            return y.squeeze()
        return y.reshape(out_shape)
