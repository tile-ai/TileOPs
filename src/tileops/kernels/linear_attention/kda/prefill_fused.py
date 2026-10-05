"""Kimi Delta Attention prefill with the chunk step and the scan in one CTA."""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    KimiDeltaAttentionCall,
    KimiDeltaAttentionFwdInterface,
)
from tileops.kernels.linear_attention.kda.fused_program import fused_chunk_program
from tileops.kernels.linear_attention.kda.prefill import CHUNK_SIZE, packed_offsets

__all__ = ["KimiDeltaAttentionFusedPrefillFwdKernel"]


class KimiDeltaAttentionFusedPrefillFwdKernel(Kernel, KimiDeltaAttentionFwdInterface):
    """SM90 prefill that keeps a chunk's whole step inside one CTA.

    The chunk-local work and the recurrence share a block, so the WY vectors,
    the gated query and key and the intra-chunk attention never reach global
    memory. The launch is one block per (sequence, value head), so this is the
    right shape only when there are enough of those to cover the device; below
    that the chunk-parallel split serves the same call.
    """

    supported_archs = [90]
    preferred_over = frozenset({"kimi_delta_attention_chunk_prefill"})

    @classmethod
    def applies(cls, call: KimiDeltaAttentionCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: KimiDeltaAttentionCall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does."""
        chunked = call.chunk_refusal
        if chunked is not None:
            return chunked
        if call.seq_len < 2:
            return "serves a sequence of at least two tokens"
        # Measured on H200: the fused shape wins from about three quarters of
        # the SMs covered and loses badly below half.
        blocks = call.sequences * call.value_heads
        wanted = call.sm_count * 3 // 4
        if blocks < wanted:
            return f"fuses the scan into {blocks} blocks, fewer than the {wanted} it needs"
        return None

    @classmethod
    def entry_for(cls, call: KimiDeltaAttentionCall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (
            call.batch,
            call.seq_len,
            call.sequences,
            call.heads,
            call.value_heads,
            call.dim_k,
            call.dim_v,
            call.scale,
            call.l2norm,
            call.dtype,
            index,
        )
        return identity, lambda: cls(
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
        """Hold the call facts the fused program is built from."""
        super().__init__(device_index=device_index)
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
        """Run every chunk inside its own CTA; see the interface for the tensors."""
        del A_log, dt_bias, cu_seqlens_cpu
        self._require_cuda(q=q, k=k, v=v, g=g, beta=beta)
        batch, seq_len = q.shape[:2]
        H, K = q.shape[2], q.shape[3]
        HV, V = v.shape[2], v.shape[3]
        total = batch * seq_len
        offsets = packed_offsets(batch, seq_len, cu_seqlens, q.device)
        num_seqs = offsets.numel() - 1

        qf = q.reshape(1, total, H, K)
        kf = k.reshape(1, total, H, K)
        vf = v.reshape(1, total, HV, V)
        gf = g.reshape(1, total, HV, K)
        bf = beta.reshape(1, total, HV)
        state = (
            torch.zeros(num_seqs, HV, K, V, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state
        )
        program = fused_chunk_program(
            H,
            HV,
            K,
            V,
            CHUNK_SIZE,
            self.dtype_to_str(self.dtype),
            self.scale,
            self.l2norm,
            total,
            num_seqs,
        )
        o, final_state = program(qf, kf, vf, gf, bf, state, offsets)
        return o.reshape(batch, seq_len, HV, V), final_state
