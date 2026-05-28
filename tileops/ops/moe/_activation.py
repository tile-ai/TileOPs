"""Activation registry for MoE expert pipelines."""
from __future__ import annotations

import torch

from tileops.ops.elementwise import GeluAndMulFwdOp, SiluAndMulFwdOp

__all__ = ["build_activation_op"]

_ACTIVATION_REGISTRY: dict[str, type] = {
    "silu_and_mul": SiluAndMulFwdOp,
    "gelu_and_mul": GeluAndMulFwdOp,
}


def build_activation_op(
    activation: str,
    M: int,
    N: int,
    dtype: torch.dtype,
    kernel_map=None,
):
    """Construct the activation sub-Op for an MoE experts pipeline.

    Args:
        activation: One of the keys in ``_ACTIVATION_REGISTRY``.
        M: Row count of the (M, 2N) gate_up tensor.
        N: Half column dim — the activation output width (= ffn_size).
        dtype: Activation/output dtype.
        kernel_map: Forwarded to the inner activation op for kernel dispatch.

    Raises:
        ValueError: If ``activation`` is not a registered key.
    """
    if activation not in _ACTIVATION_REGISTRY:
        allowed = ", ".join(sorted(_ACTIVATION_REGISTRY))
        raise ValueError(
            f"activation must be one of [{allowed}], got {activation!r}"
        )
    return _ACTIVATION_REGISTRY[activation](
        M=M, N=N, dtype=dtype, kernel_map=kernel_map,
    )
