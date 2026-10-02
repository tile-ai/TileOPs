"""Chunked Kimi Delta Attention prefill: chunk-local work, then one scan."""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    KimiDeltaAttentionCall,
    KimiDeltaAttentionFwdInterface,
)
from tileops.kernels.linear_attention.kda.chunk_programs import (
    chunk_prepare_program,
    chunk_scan_program,
)
from tileops.kernels.linear_attention.kda.packing import chunk_metadata, sequence_lengths

__all__ = ["KimiDeltaAttentionChunkPrefillFwdKernel"]

CHUNK_SIZE = 64


class KimiDeltaAttentionChunkPrefillFwdKernel(Kernel, KimiDeltaAttentionFwdInterface):
    """SM90 prefill over a 64-token chunk, equal-length or packed varlen.

    The chunk-local half runs one CTA per (chunk, value head) and the scan one
    CTA per (sequence, value head); the two meet in a workspace holding the WY
    vectors, the gated query and key, the intra-chunk attention and the chunk
    decay.
    """

    supported_archs = [90]

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
        return None

    @classmethod
    def entry_for(cls, call: KimiDeltaAttentionCall) -> Entry:
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
        del A_log, dt_bias
        self._require_cuda(q=q, k=k, v=v, g=g, beta=beta)
        batch, seq_len = q.shape[:2]
        H, K = q.shape[2], q.shape[3]
        HV, V = v.shape[2], v.shape[3]
        lengths = sequence_lengths(batch, seq_len, cu_seqlens, cu_seqlens_cpu)
        total = batch * seq_len
        num_seqs = len(lengths)
        num_chunks = sum((length + CHUNK_SIZE - 1) // CHUNK_SIZE for length in lengths)
        chunk_bos, chunk_len, seq_bos, seq_lens, seq_chunk0 = chunk_metadata(
            lengths, CHUNK_SIZE, q.device.index
        )

        flat = lambda tensor, width: tensor.reshape(1, total, tensor.shape[2], width)  # noqa: E731
        qf, kf = flat(q, K), flat(k, K)
        vf, gf = flat(v, V), flat(g, K)
        bf = beta.reshape(1, total, HV)
        state = (
            torch.zeros(num_seqs, HV, K, V, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state
        )

        name = self.dtype_to_str(self.dtype)
        prepare = chunk_prepare_program(
            H, HV, K, V, CHUNK_SIZE, name, self.scale, self.l2norm, total, num_chunks
        )
        scan = chunk_scan_program(HV, K, V, CHUNK_SIZE, name, total, num_chunks, num_seqs)
        w, u, qg, kg, aqk, dec = prepare(qf, kf, vf, gf, bf, chunk_bos, chunk_len)
        o, final_state = scan(w, u, qg, kg, aqk, dec, state, seq_bos, seq_lens, seq_chunk0)
        return o.reshape(batch, seq_len, HV, V), final_state
