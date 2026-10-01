"""GLA prefill over packed sequences, and over a batch whose rows are not a whole chunk.

A contiguous ``[batch, seq_len, heads, dim]`` tensor is the same bytes as a packed
``[1, batch * seq_len, heads, dim]`` one whose sequence offsets step by ``seq_len``, so an
equal-length call with a ragged ``seq_len`` is a packed call and runs the same four passes.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.gla.call_spec import (
    GLAInferenceCallSpec,
    GLAInferenceFwdInterface,
    build_entry,
    serves_extents,
)
from tileops.kernels.linear_attention.v_tile import GEMM_MIN_N

__all__ = ["GLAVarlenPrefillFwdKernel"]

# The tokens one chunk of the recurrence contracts over. Every pass is laid out on it, and
# it is the width the state GEMM tiles.
CHUNK_TOKENS = 64

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@functools.lru_cache(maxsize=32)
def gla_varlen_cumsum_kernel(total_tokens: int, num_seqs: int, heads: int, dim_k: int, dtype: str):
    """Accumulate the log gate inside each chunk, restarting at every sequence start.

    A token past its sequence's end contributes a zero gate, so the accumulated row the
    state pass reads at the end of a short chunk is the last real token's.
    """
    tiling = GroupTiling(num_seqs, CHUNK_TOKENS)
    num_chunks = tiling.tile_upper_bound(total_tokens)

    @tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
    def _fn(threads: int = 128):
        @T.prim_func
        def _main(
            g: T.Tensor([1, total_tokens, heads, dim_k], dtype),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            g_cumsum: T.Tensor([1, total_tokens, heads, dim_k], "float32"),
        ):
            with T.Kernel(num_chunks, heads, threads=threads) as (chunk, i_h):
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                gate = T.alloc_shared([CHUNK_TOKENS, dim_k], dtype)
                total = T.alloc_shared([CHUNK_TOKENS, dim_k], "float32")

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                if chunk < tile_cum[num_seqs]:
                    tiling.decode(chunk, tile_cum, lo, hi, seq, first)
                    start = T.cast(cu_seqlens[seq[0]], "int32") + first[0]
                    end = T.cast(cu_seqlens[seq[0] + 1], "int32")

                    for i, d in T.Parallel(CHUNK_TOKENS, dim_k):
                        gate[i, d] = T.if_then_else(
                            start + i < end,
                            g[0, T.min(start + i, end - 1), i_h, d],
                            T.cast(0, dtype),
                        )
                    for d in T.Parallel(dim_k):
                        total[0, d] = T.cast(gate[0, d], "float32")
                    for i in T.Serial(1, CHUNK_TOKENS):
                        for d in T.Parallel(dim_k):
                            total[i, d] = total[i - 1, d] + T.cast(gate[i, d], "float32")
                    for i, d in T.Parallel(CHUNK_TOKENS, dim_k):
                        if start + i < end:
                            g_cumsum[0, start + i, i_h, d] = total[i, d]

        return _main

    return _fn


@functools.lru_cache(maxsize=32)
def gla_varlen_state_kernel(
    total_tokens: int,
    num_seqs: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    num_k_partitions: int,
    num_v_partitions: int,
    num_stages: int,
):
    """Walk one sequence's chunks in order, publishing the state each chunk starts from.

    The walk's trip count is the sequence's own chunk count, so a short sequence's block
    retires early rather than stepping over the longest sequence's chunks.

    The published states carry the activation dtype: they exist only for the output pass,
    which contracts them on tensor cores and so casts them anyway. The state the caller is
    returned stays float32, which is the dtype the recurrence carries.
    """
    tiling = GroupTiling(num_seqs, CHUNK_TOKENS)
    num_chunks = tiling.tile_upper_bound(total_tokens)
    dim_k_part = dim_k // num_k_partitions
    dim_v_part = dim_v // num_v_partitions
    num_partitions = num_k_partitions * num_v_partitions
    if dim_v_part < GEMM_MIN_N:
        raise ValueError(
            f"dim_v ({dim_v}) split across num_v_partitions ({num_v_partitions}) gives a "
            f"{dim_v_part}-column T.gemm B operand, below the minimum N extent ({GEMM_MIN_N})"
        )

    @tilelang.jit(out_idx=[-2, -1], pass_configs=_PASS_CONFIGS)
    def _fn(threads: int = 128):
        @T.prim_func
        def _main(
            k: T.Tensor([1, total_tokens, heads, dim_k], dtype),
            v: T.Tensor([1, total_tokens, heads, dim_v], dtype),
            g_cumsum: T.Tensor([1, total_tokens, heads, dim_k], "float32"),
            initial_state: T.Tensor([num_seqs, heads, dim_k, dim_v], "float32"),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            chunk_state: T.Tensor([num_chunks, heads, dim_k, dim_v], dtype),
            final_state: T.Tensor([num_seqs, heads, dim_k, dim_v], "float32"),
        ):
            with T.Kernel(num_seqs * num_partitions, heads, threads=threads) as (bx, i_h):
                seq = bx // num_partitions
                k_offset = (bx % num_partitions) // num_v_partitions * dim_k_part
                v_offset = (bx % num_partitions) % num_v_partitions * dim_v_part

                tile_cum = T.alloc_shared([num_seqs + 1], "int32")
                # The state is the GEMM's own accumulator: decaying it and adding the
                # chunk's update in place keeps it off shared memory between chunks.
                state = T.alloc_fragment([dim_k_part, dim_v_part], "float32")
                keys = T.alloc_shared([CHUNK_TOKENS, dim_k_part], dtype)
                values = T.alloc_shared([CHUNK_TOKENS, dim_v_part], dtype)
                gate = T.alloc_shared([CHUNK_TOKENS, dim_k_part], "float32")
                decayed = T.alloc_fragment([CHUNK_TOKENS, dim_k_part], dtype)
                last = T.alloc_fragment([dim_k_part], "float32")

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                start = T.cast(cu_seqlens[seq], "int32")
                end = T.cast(cu_seqlens[seq + 1], "int32")
                base = tile_cum[seq]

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    state[i_k, i_v] = initial_state[seq, i_h, k_offset + i_k, v_offset + i_v]

                # The trip count comes from the offsets rather than from ``tile_cum``: a
                # shared-memory read cannot stand in a loop extent.
                for i_c in T.Pipelined(T.ceildiv(end - start, CHUNK_TOKENS), num_stages=num_stages):
                    first = start + i_c * CHUNK_TOKENS
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        chunk_state[base + i_c, i_h, k_offset + i_k, v_offset + i_v] = T.cast(
                            state[i_k, i_v], dtype
                        )
                    # A chunk inside the sequence is copied as a tile, which is what the
                    # prefetch above pipelines; only the last chunk of a sequence takes the
                    # predicated path, where a token past the end reads as a zero key and
                    # value and repeats the last accumulated gate.
                    if first + CHUNK_TOKENS <= end:
                        T.copy(
                            k[
                                0,
                                first : first + CHUNK_TOKENS,
                                i_h,
                                k_offset : k_offset + dim_k_part,
                            ],
                            keys,
                            disable_tma=True,
                        )
                        T.copy(
                            v[
                                0,
                                first : first + CHUNK_TOKENS,
                                i_h,
                                v_offset : v_offset + dim_v_part,
                            ],
                            values,
                            disable_tma=True,
                        )
                        T.copy(
                            g_cumsum[
                                0,
                                first : first + CHUNK_TOKENS,
                                i_h,
                                k_offset : k_offset + dim_k_part,
                            ],
                            gate,
                            disable_tma=True,
                        )
                    else:
                        for i, d in T.Parallel(CHUNK_TOKENS, dim_k_part):
                            gate[i, d] = g_cumsum[0, T.min(first + i, end - 1), i_h, k_offset + d]
                            keys[i, d] = T.if_then_else(
                                first + i < end,
                                k[0, T.min(first + i, end - 1), i_h, k_offset + d],
                                T.cast(0, dtype),
                            )
                        for i, d in T.Parallel(CHUNK_TOKENS, dim_v_part):
                            values[i, d] = T.if_then_else(
                                first + i < end,
                                v[0, T.min(first + i, end - 1), i_h, v_offset + d],
                                T.cast(0, dtype),
                            )

                    for d in T.Parallel(dim_k_part):
                        last[d] = gate[CHUNK_TOKENS - 1, d]
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        state[i_k, i_v] = state[i_k, i_v] * T.exp2(last[i_k] * LOG2E)
                    for i, d in T.Parallel(CHUNK_TOKENS, dim_k_part):
                        decayed[i, d] = T.cast(
                            T.cast(keys[i, d], "float32") * T.exp2((last[d] - gate[i, d]) * LOG2E),
                            dtype,
                        )
                    T.gemm(
                        decayed, values, state, transpose_A=True, policy=T.GemmWarpPolicy.FullRow
                    )

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    final_state[seq, i_h, k_offset + i_k, v_offset + i_v] = state[i_k, i_v]

        return _main

    return _fn


@functools.lru_cache(maxsize=32)
def gla_varlen_causal_kernel(
    total_tokens: int,
    num_seqs: int,
    heads: int,
    dim_k: int,
    scale: float,
    dtype: str,
):
    """Form the causal query-key product of one chunk, one 16-token sub-block at a time.

    Every pair anchors both exponents on the query sub-block's first row and runs on tensor
    cores, the diagonal one included: within one sub-block the two exponents travel at most
    its own 16 tokens of gate, which float32 holds, and the causal mask is applied to the
    product rather than to the operands.
    """
    # The sub-block the chunk's causal product is tiled by. It bounds how far the two
    # exponents around one anchor row can travel, which is what keeps them representable.
    SUBCHUNK_TOKENS = 16

    tiling = GroupTiling(num_seqs, CHUNK_TOKENS)
    num_chunks = tiling.tile_upper_bound(total_tokens)
    num_subchunks = CHUNK_TOKENS // SUBCHUNK_TOKENS

    @tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
    def _fn(threads: int = 128):
        @T.prim_func
        def _main(
            q: T.Tensor([1, total_tokens, heads, dim_k], dtype),
            k: T.Tensor([1, total_tokens, heads, dim_k], dtype),
            g_cumsum: T.Tensor([1, total_tokens, heads, dim_k], "float32"),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            causal: T.Tensor([1, total_tokens, heads, CHUNK_TOKENS], dtype),
        ):
            with T.Kernel(num_chunks * num_subchunks * num_subchunks, heads, threads=threads) as (
                bx,
                i_h,
            ):
                chunk = bx // (num_subchunks * num_subchunks)
                bi = bx % (num_subchunks * num_subchunks) // num_subchunks
                bj = bx % num_subchunks

                tile_cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                queries = T.alloc_shared([SUBCHUNK_TOKENS, dim_k], dtype)
                keys = T.alloc_shared([SUBCHUNK_TOKENS, dim_k], dtype)
                g_q = T.alloc_shared([SUBCHUNK_TOKENS, dim_k], "float32")
                g_k = T.alloc_shared([SUBCHUNK_TOKENS, dim_k], "float32")
                block = T.alloc_shared([SUBCHUNK_TOKENS, SUBCHUNK_TOKENS], dtype)
                q_gated = T.alloc_shared([SUBCHUNK_TOKENS, dim_k], dtype)
                k_gated = T.alloc_shared([SUBCHUNK_TOKENS, dim_k], dtype)
                product = T.alloc_fragment([SUBCHUNK_TOKENS, SUBCHUNK_TOKENS], "float32")

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                if chunk < tile_cum[num_seqs]:
                    tiling.decode(chunk, tile_cum, lo, hi, seq, first)
                    start = T.cast(cu_seqlens[seq[0]], "int32") + first[0]
                    end = T.cast(cu_seqlens[seq[0] + 1], "int32")
                    row = start + bi * SUBCHUNK_TOKENS
                    col = start + bj * SUBCHUNK_TOKENS

                    if bj <= bi:
                        for i, d in T.Parallel(SUBCHUNK_TOKENS, dim_k):
                            queries[i, d] = q[0, T.min(row + i, end - 1), i_h, d]
                            g_q[i, d] = g_cumsum[0, T.min(row + i, end - 1), i_h, d]
                        for j, d in T.Parallel(SUBCHUNK_TOKENS, dim_k):
                            keys[j, d] = T.if_then_else(
                                col + j < end,
                                k[0, T.min(col + j, end - 1), i_h, d],
                                T.cast(0, dtype),
                            )
                            g_k[j, d] = g_cumsum[0, T.min(col + j, end - 1), i_h, d]

                        for i, d in T.Parallel(SUBCHUNK_TOKENS, dim_k):
                            q_gated[i, d] = T.cast(
                                T.cast(queries[i, d], "float32")
                                * T.exp2((g_q[i, d] - g_q[0, d]) * LOG2E)
                                * scale,
                                dtype,
                            )
                        for j, d in T.Parallel(SUBCHUNK_TOKENS, dim_k):
                            k_gated[j, d] = T.cast(
                                T.cast(keys[j, d], "float32")
                                * T.exp2((g_q[0, d] - g_k[j, d]) * LOG2E),
                                dtype,
                            )
                        T.fill(product, 0.0)
                        T.gemm(q_gated, k_gated, product, transpose_B=True)
                        for i, j in T.Parallel(SUBCHUNK_TOKENS, SUBCHUNK_TOKENS):
                            block[i, j] = T.cast(
                                T.if_then_else(bj < bi or j <= i, product[i, j], 0.0), dtype
                            )
                    else:
                        for i, j in T.Parallel(SUBCHUNK_TOKENS, SUBCHUNK_TOKENS):
                            block[i, j] = T.cast(0, dtype)

                    for i, j in T.Parallel(SUBCHUNK_TOKENS, SUBCHUNK_TOKENS):
                        if row + i < end:
                            causal[0, row + i, i_h, bj * SUBCHUNK_TOKENS + j] = block[i, j]

        return _main

    return _fn


@functools.lru_cache(maxsize=32)
def gla_varlen_output_kernel(
    total_tokens: int,
    num_seqs: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    scale: float,
    dtype: str,
):
    """Read the chunk's own state and causal product into the two output contractions."""
    tiling = GroupTiling(num_seqs, CHUNK_TOKENS)
    num_chunks = tiling.tile_upper_bound(total_tokens)

    @tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
    def _fn(threads: int = 128):
        @T.prim_func
        def _main(
            q: T.Tensor([1, total_tokens, heads, dim_k], dtype),
            v: T.Tensor([1, total_tokens, heads, dim_v], dtype),
            g_cumsum: T.Tensor([1, total_tokens, heads, dim_k], "float32"),
            chunk_state: T.Tensor([num_chunks, heads, dim_k, dim_v], dtype),
            causal: T.Tensor([1, total_tokens, heads, CHUNK_TOKENS], dtype),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            o: T.Tensor([1, total_tokens, heads, dim_v], dtype),
        ):
            with T.Kernel(num_chunks, heads, threads=threads) as (chunk, i_h):
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                queries = T.alloc_shared([CHUNK_TOKENS, dim_k], dtype)
                values = T.alloc_shared([CHUNK_TOKENS, dim_v], dtype)
                gate = T.alloc_shared([CHUNK_TOKENS, dim_k], "float32")
                weights = T.alloc_shared([CHUNK_TOKENS, CHUNK_TOKENS], dtype)
                q_gated = T.alloc_shared([CHUNK_TOKENS, dim_k], dtype)
                state = T.alloc_shared([dim_k, dim_v], dtype)
                acc = T.alloc_fragment([CHUNK_TOKENS, dim_v], "float32")

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                if chunk < tile_cum[num_seqs]:
                    tiling.decode(chunk, tile_cum, lo, hi, seq, first)
                    start = T.cast(cu_seqlens[seq[0]], "int32") + first[0]
                    end = T.cast(cu_seqlens[seq[0] + 1], "int32")

                    for i, d in T.Parallel(CHUNK_TOKENS, dim_k):
                        queries[i, d] = q[0, T.min(start + i, end - 1), i_h, d]
                        gate[i, d] = g_cumsum[0, T.min(start + i, end - 1), i_h, d]
                    for i, d in T.Parallel(CHUNK_TOKENS, dim_v):
                        values[i, d] = v[0, T.min(start + i, end - 1), i_h, d]
                    for i, j in T.Parallel(CHUNK_TOKENS, CHUNK_TOKENS):
                        weights[i, j] = causal[0, T.min(start + i, end - 1), i_h, j]
                    for d, j in T.Parallel(dim_k, dim_v):
                        state[d, j] = chunk_state[chunk, i_h, d, j]
                    for i, d in T.Parallel(CHUNK_TOKENS, dim_k):
                        q_gated[i, d] = T.cast(
                            T.cast(queries[i, d], "float32") * T.exp2(gate[i, d] * LOG2E), dtype
                        )

                    T.fill(acc, 0.0)
                    T.gemm(q_gated, state, acc)
                    for i, j in T.Parallel(CHUNK_TOKENS, dim_v):
                        acc[i, j] *= scale
                    T.gemm(weights, values, acc)
                    for i, j in T.Parallel(CHUNK_TOKENS, dim_v):
                        if start + i < end:
                            o[0, start + i, i_h, j] = T.cast(acc[i, j], dtype)

        return _main

    return _fn


class GLAVarlenPrefillFwdKernel(Kernel, GLAInferenceFwdInterface):
    """Run the chunked recurrence per sequence, with every launch bounded by the shapes.

    Each chunk-parallel launch covers ``total_tokens // 64 + num_sequences`` chunks, the most
    the offsets can describe, and a block past the count the offsets give retires at once.
    """

    supported_archs = [90]

    # Threads per block. The state walk and the causal product run two warps: their GEMMs
    # tile a 16-wide operand, which a wider block cannot partition into whole warps. The
    # gate accumulation and the output contractions run eight, which is what their
    # full-width tiles fill. Re-fit by timing the manifest rows at 64, 128 and 256.
    _state_threads = 64
    _wide_threads = 256

    # Stages the state walk prefetches a chunk's operands over. The walk is serial, so
    # without them every chunk's GEMM waits on its own loads. Re-fit with the manifest rows
    # over 1 to 4; more stages cost shared memory the state tile also needs.
    _state_stages = 2

    @classmethod
    def applies(cls, call: GLAInferenceCallSpec) -> bool:
        """Either the call is packed, or its rows are equal-length and not a whole chunk.

        The two are the same work once the rows are read as one packed sequence each.
        """
        return serves_extents(call) and (
            call.varlen or (call.seq_len > 1 and call.seq_len % CHUNK_TOKENS != 0)
        )

    @classmethod
    def entry_for(cls, call: GLAInferenceCallSpec) -> Entry:
        # What the launch waits on is one state block's walk, so the state tile is split
        # until the device is covered. The key axis is split first and all the way: a block
        # then reads only its own key channels and its own slice of the float32 gate, and
        # nothing is read twice. Splitting the value axis reads those two again per slice,
        # so it is taken only while it still puts another block on an idle multiprocessor.
        k_partitions = call.dim_k // GEMM_MIN_N
        v_partitions = 1
        blocks = call.num_sequences * call.heads * k_partitions
        while (
            blocks * v_partitions * 2 <= call.sm_count
            and call.dim_v // (v_partitions * 2) >= GEMM_MIN_N
        ):
            v_partitions *= 2
        return build_entry(
            cls,
            call,
            batch=call.batch,
            seq_len=call.seq_len,
            num_sequences=call.num_sequences,
            heads=call.heads,
            dim_k=call.dim_k,
            dim_v=call.dim_v,
            k_partitions=k_partitions,
            v_partitions=v_partitions,
        )

    def __init__(
        self,
        batch: int,
        seq_len: int,
        num_sequences: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        k_partitions: int,
        v_partitions: int,
        scale: float,
        dtype: torch.dtype,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        total = batch * seq_len
        name = self.dtype_to_str(dtype)
        self._cumsum = gla_varlen_cumsum_kernel(total, num_sequences, heads, dim_k, name)(
            self._wide_threads
        )
        self._state = gla_varlen_state_kernel(
            total,
            num_sequences,
            heads,
            dim_k,
            dim_v,
            name,
            k_partitions,
            v_partitions,
            self._state_stages,
        )(self._state_threads)
        self._causal = gla_varlen_causal_kernel(total, num_sequences, heads, dim_k, scale, name)(
            self._state_threads
        )
        self._output = gla_varlen_output_kernel(
            total, num_sequences, heads, dim_k, dim_v, scale, name
        )(self._wide_threads)

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
        # The offsets are read on the device, so the host copy is never needed.
        del cu_seqlens_cpu
        if cu_seqlens is None:
            # Built per call rather than held: a tensor allocated on one stream and reused
            # on another is read before its own write unless the second stream is made to
            # wait, and an equal-length call's offsets cost one launch of ``batch + 1``.
            cu_seqlens = torch.arange(
                0,
                (self.batch + 1) * self.seq_len,
                self.seq_len,
                dtype=torch.int64,
                device=q.device,
            )
        packed = (1, self.batch * self.seq_len, self.heads, -1)
        state = (
            torch.zeros(
                self.num_sequences,
                self.heads,
                self.dim_k,
                self.dim_v,
                dtype=torch.float32,
                device=q.device,
            )
            if initial_state is None
            else initial_state
        )
        gate = self._cumsum(g.view(packed), cu_seqlens)
        chunk_state, final_state = self._state(
            k.view(packed), v.view(packed), gate, state, cu_seqlens
        )
        causal = self._causal(q.view(packed), k.view(packed), gate, cu_seqlens)
        o = self._output(q.view(packed), v.view(packed), gate, chunk_state, causal, cu_seqlens)
        return o.view(self.batch, self.seq_len, self.heads, self.dim_v), final_state
