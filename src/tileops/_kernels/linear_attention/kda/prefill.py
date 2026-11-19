"""Chunked Kimi Delta Attention (KDA) prefill: chunk-local work, then one scan."""

import torch

from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.linear_attention.call_spec import (
    KDACall,
    KDAFwdInterface,
)
from tileops._kernels.linear_attention.chunk_plan import launch_plan
from tileops._kernels.linear_attention.chunk_prefill import run_chunk_prefill

__all__ = ["KDAChunkPrefillFwdKernel"]


class KDAChunkPrefillFwdKernel(Kernel, KDAFwdInterface):
    """Prefill over a 64-token chunk, equal-length or packed varlen.

    The chunk-local half runs one CTA per (chunk, value head) and the scan one
    CTA per (sequence, value head); the two meet in a workspace holding the WY
    vectors, the gated query and key, the intra-chunk attention and the chunk
    decay. Where shared memory holds less, the chunk-local half takes its lean
    form and the scan splits the state's value columns across CTAs. The scan reads
    and writes the states in either layout, in batch order or in the slots of a pool.
    """

    supported_archs = [80, 89, 90]

    @classmethod
    def refusal(cls, call: KDACall) -> str | None:
        """Why this kernel does not serve *call*, or ``None`` when it does."""
        chunked = call.chunk_refusal
        if chunked is not None:
            return chunked
        if call.seq_len < 2:
            return "serves a sequence of at least two tokens"
        plan = launch_plan(
            call.dim_k,
            call.dtype.itemsize,
            call.sequences,
            call.smem_budget,
            call.arch,
            call.l2norm,
        )
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
            call.gate_in_kernel,
            call.has_dt_bias,
            call.lower_bound,
            call.varlen,
            call.state_v_first,
            call.pooled,
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
            gate_in_kernel=call.gate_in_kernel,
            has_dt_bias=call.has_dt_bias,
            lower_bound=call.lower_bound,
            dtype=call.dtype,
            device_index=index,
            state_v_first=call.state_v_first,
            pooled=call.pooled,
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
        device_index: int | None = None,
        gate_in_kernel: bool = False,
        has_dt_bias: bool = False,
        lower_bound: float | None = None,
        state_v_first: bool = False,
        pooled: bool = False,
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
        self.gate_in_kernel = gate_in_kernel
        self.has_dt_bias = has_dt_bias
        self.lower_bound = lower_bound
        self.dtype = dtype
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
        """Run the chunk-local half, then the scan; see the interface for the tensors."""
        del cu_seqlens_cpu
        return chunk_prefill(
            self, q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias, state_indices
        )


def chunk_prefill(
    kernel, q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias, state_indices
) -> tuple[torch.Tensor, torch.Tensor]:
    """The chunk-local half, then the scan, on the programs *kernel* holds and its recurrence
    facts (``scale``, ``l2norm``, ``dtype``, the gate facts and the state layout)."""
    kernel._require_devices(q=q, k=k, v=v, g=g, beta=beta)
    batch, seq_len = q.shape[:2]
    H, K = q.shape[2], q.shape[3]
    HV, V = v.shape[2], v.shape[3]
    # One kernel serves every sequence count and initial-state presence its
    # identity leaves open, so the call the driver reads restates them from this
    # call's tensors; the variants `refusal` declines stay at their defaults.
    call = KDACall(
        batch=batch,
        seq_len=seq_len,
        sequences=batch if cu_seqlens is None else cu_seqlens.shape[0] - 1,
        has_initial_state=initial_state is not None,
        varlen=cu_seqlens is not None,
        heads=H,
        value_heads=HV,
        dim_k=K,
        dim_v=V,
        dtype=kernel.dtype,
        scale=kernel.scale,
        l2norm=kernel.l2norm,
        gate_in_kernel=kernel.gate_in_kernel,
        has_dt_bias=kernel.has_dt_bias,
        lower_bound=kernel.lower_bound,
        state_v_first=kernel.state_v_first,
        pooled=kernel.pooled,
        device=q.device,
    )
    inputs = (q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias)
    return run_chunk_prefill(kernel, call, inputs, gate_mode="channel", slots=state_indices)
