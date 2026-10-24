"""Kimi Delta Attention (KDA) prefill with the chunk step and the scan in one CTA."""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    KDACall,
    KDAFwdInterface,
)
from tileops.kernels.linear_attention.chunk_plan import CHUNK_SIZE, packed_offsets
from tileops.kernels.linear_attention.kda.fused_program import fused_chunk_program

__all__ = ["KDAFusedPrefillFwdKernel"]

# Tokens times value heads per SM from which an SM90 call with precomputed gates takes two
# launches: checking each chunk costs a share of every step, a second launch a fixed switch
# that only enough tokens per SM amortize. The rerun test sizes its call from it.
TWO_LAUNCH_TOKENS_PER_SM = 3072


class KDAFusedPrefillFwdKernel(Kernel, KDAFwdInterface):
    """Prefill that keeps a chunk's whole step inside one CTA.

    The chunk-local work and the recurrence share a block, so the WY vectors,
    the gated query and key and the intra-chunk attention never reach global
    memory. The launch is one block per (sequence, value head), so this is the
    right shape only when there are enough of those to cover the device; below
    that the chunk-parallel split serves the same call.
    """

    supported_archs = [89, 90]
    preferred_over = frozenset({"kda_chunk_prefill"})

    @classmethod
    def refusal(cls, call: KDACall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does."""
        chunked = call.chunk_refusal
        if chunked is not None:
            return chunked
        if call.seq_len < 2:
            return "serves a sequence of at least two tokens"
        if call.gate_in_kernel:
            return "raw gates use chunk-parallel preparation"
        # A CTA per (sequence, value head) pays off only while those cover most SMs; a
        # compact 128-wide SM89 CTA fills one SM's shared memory.
        blocks = call.sequences * call.value_heads
        wanted = call.sm_count if call.arch == 89 and call.dim_k == 128 else call.sm_count * 3 // 4
        if blocks < wanted:
            return f"fuses the scan into {blocks} blocks, fewer than the {wanted} it needs"
        return None

    @classmethod
    def entry_for(cls, call: KDACall) -> Entry:
        index = call.device.index if call.device is not None else None
        compact = call.arch == 89 and call.dim_k == 128
        two_launch = (
            call.arch == 90
            and call.batch * call.seq_len * call.value_heads
            >= TWO_LAUNCH_TOKENS_PER_SM * call.sm_count
        )
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
            compact,
            two_launch,
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
            compact=compact,
            two_launch=two_launch,
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
        compact: bool = False,
        two_launch: bool = False,
    ) -> None:
        """Hold the call facts the fused program is built from."""
        super().__init__(device_index=device_index)
        self.compact = compact
        self.heads = heads
        self.value_heads = value_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale
        self.l2norm = l2norm
        self.dtype = dtype
        self.two_launch = two_launch

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
        del cu_seqlens_cpu, A_log, dt_bias
        self._require_devices(q=q, k=k, v=v, g=g, beta=beta)
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
        shape = (
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
        if self.two_launch:
            o, final_state, flags = fused_chunk_program(*shape, stage="check")(
                qf, kf, vf, gf, bf, state, offsets
            )
            fused_chunk_program(*shape, stage="rerun")(
                qf, kf, vf, gf, bf, state, offsets, o, final_state, flags
            )
        else:
            program = fused_chunk_program(*shape, compact=self.compact)
            o, final_state = program(qf, kf, vf, gf, bf, state, offsets)
        return o.reshape(batch, seq_len, HV, V), final_state
