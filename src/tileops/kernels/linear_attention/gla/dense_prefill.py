"""Inference-owned dense GLA prefill kernel."""

from typing import Optional

import torch

from .gla_fwd import GLAFwdKernel

__all__ = ["GLADensePrefillFwdKernel"]


class GLADensePrefillFwdKernel(GLAFwdKernel):
    """Run the proven chunkwise GPU programs and always return FP32 state.

    Keeping this entry separate from the training forward lets inference
    specialize independently without changing the training or decode kernels.
    """

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        scale: float,
        dtype: torch.dtype,
        device_index: Optional[int] = None,
    ) -> None:
        del device_index  # Part of the Op cache identity; the parent uses tensor devices.
        super().__init__(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=dim_v,
            chunk_size=64,
            scale=scale,
            output_final_state=True,
            dtype=dtype,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is not None or cu_seqlens_cpu is not None:
            raise ValueError("the in-tree GLA dense-prefill kernel does not support packed varlen")
        o, final_state = super().forward(q, k, v, g, initial_state)
        assert final_state is not None
        return o, final_state
