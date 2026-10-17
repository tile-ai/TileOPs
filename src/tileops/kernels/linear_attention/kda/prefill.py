"""Chunked Kimi Delta Attention (KDA) prefill: chunk-local work, then one scan."""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    KDACall,
    KDAFwdInterface,
)
from tileops.kernels.linear_attention.chunk_plan import _plan
from tileops.kernels.linear_attention.chunk_prefill import run_chunk_prefill

__all__ = ["KDAChunkPrefillFwdKernel"]


class KDAChunkPrefillFwdKernel(Kernel, KDAFwdInterface):
    """Prefill over a 64-token chunk, equal-length or packed varlen.

    The chunk-local half runs one CTA per (chunk, value head) and the scan one
    CTA per (sequence, value head); the two meet in a workspace holding the WY
    vectors, the gated query and key, the intra-chunk attention and the chunk
    decay. Where shared memory holds less, the chunk-local half takes its lean
    form and the scan splits the state's value columns across CTAs.
    """

    supported_archs = [80, 89, 90]

    @classmethod
    def refusal(cls, call: KDACall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does."""
        chunked = call.chunk_refusal
        if chunked is not None:
            return chunked
        if call.seq_len < 2:
            return "serves a sequence of at least two tokens"
        plan = _plan(call.dim_k, call.dtype.itemsize, call.sequences, call.smem_budget, call.arch)
        if plan is None:
            return (
                f"needs more than {call.smem_budget} bytes of shared memory for "
                f"{call.sequences} sequences"
            )
        return None

    @classmethod
    def entry_for(cls, call: KDACall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (
            call.batch,
            call.seq_len,
            call.heads,
            call.value_heads,
            call.dim_k,
            call.dim_v,
            call.scale,
            call.l2norm,
            call.varlen,
            call.dtype,
            index,
        )
        return identity, lambda: cls(
            batch=call.batch,
            seq_len=call.seq_len,
            heads=call.heads,
            value_heads=call.value_heads,
            dim_k=call.dim_k,
            dim_v=call.dim_v,
            scale=call.scale,
            l2norm=call.l2norm,
            dtype=call.dtype,
            device_index=index,
        )

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        value_heads: int,
        dim_k: int,
        dim_v: int,
        scale: float,
        l2norm: bool,
        dtype: torch.dtype,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        """Hold the call facts both programs are built from."""
        super().__init__(device_index=device_index)
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.value_heads = value_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale
        self.l2norm = l2norm
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the chunk-local half, then the scan; see the interface for the tensors."""
        del A_log, dt_bias, cu_seqlens_cpu
        self._require_devices(q=q, k=k, v=v, g=g, beta=beta)
        batch, seq_len = q.shape[:2]
        H, K = q.shape[2], q.shape[3]
        HV, V = v.shape[2], v.shape[3]
        call = KDACall(
            batch=batch,
            seq_len=seq_len,
            heads=H,
            value_heads=HV,
            dim_k=K,
            dim_v=V,
            dtype=self.dtype,
            scale=self.scale,
            l2norm=self.l2norm,
            device=q.device,
        )
        inputs = (q, k, v, g, beta, initial_state, cu_seqlens, None, None)
        return run_chunk_prefill(call, inputs, gate_mode="channel")
