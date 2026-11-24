"""Kimi Delta Attention (KDA) prefill with the chunk step and the scan in one CTA."""

import torch

from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._kernels.linear_attention.call_spec import (
    KDACall,
    KDAFwdInterface,
)
from tileops._kernels.linear_attention.chunk_plan import CHUNK_SIZE, packed_offsets
from tileops._kernels.linear_attention.chunk_prefill import referenced_form
from tileops._kernels.linear_attention.kda.fused_program import fused_chunk_program
from tileops._kernels.linear_attention.kda.prefill import KDAChunkPrefillFwdKernel

__all__ = ["KDAFusedPrefillFwdKernel"]


class KDAFusedPrefillFwdKernel(Kernel, KDAFwdInterface):
    """Prefill that keeps a chunk's whole step inside one CTA.

    The chunk-local work and the recurrence share a block, so the WY vectors,
    the gated query and key and the intra-chunk attention never reach global
    memory. The launch is one block per (sequence, value head).
    """

    supported_archs = [89, 90]
    preferred_over = frozenset({"kda_chunk_prefill"})

    @classmethod
    def refusal(cls, call: KDACall) -> str | None:
        """Why this kernel does not serve *call*, or ``None`` when it does."""
        chunked = call.chunk_refusal
        if chunked is not None:
            return chunked
        if call.seq_len < 2:
            return "serves a sequence of at least two tokens"
        if call.gate_in_kernel:
            return "raw gates use chunk-parallel preparation"
        # On SM90 the chunk-parallel path prepares every chunk at once and walks only the
        # states, while this CTA walks its sequence's chunks whole, each chunk's preparation on
        # the walk's serial chain. Given gates of FP32's exponent range take the joint launch.
        # Other given gates take the split launches, which outrun this CTA unless the call holds
        # at most one chunk a sequence on average and its CTAs run in one wave: then this
        # launch, too, prepares every chunk at once, in one launch instead of several.
        if call.arch == 90 and KDAChunkPrefillFwdKernel.refusal(call) is None:
            if referenced_form(call):
                return "SM90 prepares given gates of FP32's exponent range in the joint launch"
            one_chunk = call.batch * call.seq_len <= call.sequences * CHUNK_SIZE
            one_wave = call.sequences * call.value_heads <= call.sm_count
            if not (one_chunk and one_wave):
                return "SM90 prepares chunks past one a sequence or one wave in the split launches"
        # A CTA per (sequence, value head) pays off only while those cover most SMs; a
        # compact 128-wide SM89 CTA fills one SM's shared memory.
        blocks = call.sequences * call.value_heads
        wanted = call.sm_count if call.arch == 89 and call.dim_k == 128 else call.sm_count * 3 // 4
        if blocks < wanted:
            return f"fuses the scan into {blocks} blocks, fewer than the {wanted} it needs"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: KDACall) -> Entry:
        index = call.device_index
        compact = call.arch == 89 and call.dim_k == 128
        # A second SM90 launch amortizes its switch cost above this per-SM workload. A state
        # pool takes the one-launch program: the second launch restarts a pair from its
        # initial state, which the first has overwritten in the pool.
        two_launch_tokens_per_sm = 3072
        two_launch = (
            call.arch == 90
            and not call.pooled
            and call.batch * call.seq_len * call.value_heads
            >= two_launch_tokens_per_sm * call.sm_count
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
            call.state_v_first,
            call.pooled,
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
            state_v_first=call.state_v_first,
            pooled=call.pooled,
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
        device_index: int | None = None,
        compact: bool = False,
        two_launch: bool = False,
        state_v_first: bool = False,
        pooled: bool = False,
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
        self.state_v_first = state_v_first
        self.pooled = pooled

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        # A square state: the zero state is the same in either layout.
        state = (
            torch.zeros(num_seqs, HV, K, V, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state
        )
        program = self.program(self._launcher, H, HV, K, V, total, num_seqs)
        if self.pooled:
            # The pool is advanced in place and returned as the final state.
            o = program(qf, kf, vf, gf, bf, state, offsets, state_indices)
            return o.reshape(batch, seq_len, HV, V), state
        o, final_state = program(qf, kf, vf, gf, bf, state, offsets)
        return o.reshape(batch, seq_len, HV, V), final_state

    def _launcher(self, cfg: dict, H: int, HV: int, K: int, V: int, total: int, num_seqs: int):
        """What one launch over *total* packed tokens in *num_seqs* sequences runs: the
        fused program, or the checking launch and the rerun of the flagged pairs."""
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
        layout = {"state_v_first": self.state_v_first}
        if not self.two_launch:
            return fused_chunk_program(*shape, compact=self.compact, pooled=self.pooled, **layout)
        check = CompiledProgram(fused_chunk_program(*shape, stage="check", **layout))
        rerun = CompiledProgram(fused_chunk_program(*shape, stage="rerun", **layout))

        def two_launch(qf, kf, vf, gf, bf, state, offsets):
            o, final_state, flags = check(qf, kf, vf, gf, bf, state, offsets)
            rerun(qf, kf, vf, gf, bf, state, offsets, o, final_state, flags)
            return o, final_state

        return two_launch
