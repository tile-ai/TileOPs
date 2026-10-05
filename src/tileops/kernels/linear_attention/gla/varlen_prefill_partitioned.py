"""GLA prefill over packed sequences whose state walk runs on partitions of a sequence.

The recurrence maps a chunk's incoming state to ``h * exp(gate) + update``, and that map
composes, so a sequence's chunks need not be walked in order by one block: each partition of
``partition_chunks`` chunks walks from a zero state, a scan over the partition summaries
gives the state each partition starts from, and the output pass restores a chunk's state
from the two.

What that buys is the slicing it makes unnecessary. A walk of one block per sequence covers
the device by splitting the state tile, and every key slice reads the values again while
every value slice reads the keys and the float32 gate again. Partitions cover the device
without re-reading anything, so the tile is split less and the walk moves less.

The gate accumulation and the causal product are the same two passes
:mod:`tileops.kernels.linear_attention.gla.varlen_prefill` runs; this module replaces the
state walk and the output pass that reads it.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import head_count_refusal
from tileops.kernels.linear_attention.gla.call_spec import (
    GLAInferenceCallSpec,
    GLAInferenceFwdInterface,
    build_entry,
    serves_extents,
)
from tileops.kernels.linear_attention.gla.varlen_prefill import (
    CHUNK_TOKENS,
    gla_varlen_causal_kernel,
    gla_varlen_cumsum_kernel,
)
from tileops.kernels.linear_attention.v_tile import GEMM_MIN_N

__all__ = ["GLAVarlenPrefillPartitionedFwdKernel"]

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@functools.lru_cache(maxsize=32)
def gla_varlen_local_state_kernel(
    total_tokens: int,
    num_seqs: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    num_k_partitions: int,
    num_v_partitions: int,
    partition_chunks: int,
    num_stages: int,
):
    """Walk one partition's chunks from a zero state, publishing what the scan completes.

    A partition is ``partition_chunks`` chunks of one sequence, so the chain a block walks
    is that length whatever the sequence's own length is, and the chunks of a long sequence
    become independent work instead of one block's serial walk.

    Each chunk gets its partition-local state and the gate the partition has accumulated by
    it; each partition gets the state and the gate it ends at, which is the affine summary
    the scan composes.

    The per-chunk states carry the activation dtype: they exist only for the output pass,
    which contracts them on tensor cores and so casts them anyway. The summaries stay
    float32, because the scan's recurrence is what the caller's final state comes from.
    """
    part_tiling = GroupTiling(num_seqs, CHUNK_TOKENS * partition_chunks)
    chunk_tiling = GroupTiling(num_seqs, CHUNK_TOKENS)
    num_partitions = part_tiling.tile_upper_bound(total_tokens)
    num_chunks = chunk_tiling.tile_upper_bound(total_tokens)
    dim_k_part = dim_k // num_k_partitions
    dim_v_part = dim_v // num_v_partitions
    num_slices = num_k_partitions * num_v_partitions
    if dim_v_part < GEMM_MIN_N:
        raise ValueError(
            f"dim_v ({dim_v}) split across num_v_partitions ({num_v_partitions}) gives a "
            f"{dim_v_part}-column T.gemm B operand, below the minimum N extent ({GEMM_MIN_N})"
        )

    @tilelang.jit(out_idx=[-4, -3, -2, -1], pass_configs=_PASS_CONFIGS)
    def _fn(threads: int = 128):
        @T.prim_func
        def _main(
            k: T.Tensor([1, total_tokens, heads, dim_k], dtype),
            v: T.Tensor([1, total_tokens, heads, dim_v], dtype),
            g_cumsum: T.Tensor([1, total_tokens, heads, dim_k], "float32"),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            chunk_state: T.Tensor([num_chunks, heads, dim_k, dim_v], dtype),
            chunk_reach: T.Tensor([num_chunks, heads, dim_k], "float32"),
            summary_state: T.Tensor([num_partitions, heads, dim_k, dim_v], "float32"),
            summary_reach: T.Tensor([num_partitions, heads, dim_k], "float32"),
        ):
            with T.Kernel(num_partitions * num_slices, heads, threads=threads) as (bx, i_h):
                part = bx // num_slices
                k_offset = (bx % num_slices) // num_v_partitions * dim_k_part
                v_offset = (bx % num_slices) % num_v_partitions * dim_v_part
                # ``chunk_reach`` has no value axis: one value slice of a key slice writes it.
                leads = (bx % num_slices) % num_v_partitions == 0

                part_cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                # The state is the GEMM's own accumulator, so no chunk lands it in shared
                # memory.
                state = T.alloc_fragment([dim_k_part, dim_v_part], "float32")
                reach = T.alloc_fragment([dim_k_part], "float32")
                keys = T.alloc_shared([CHUNK_TOKENS, dim_k_part], dtype)
                values = T.alloc_shared([CHUNK_TOKENS, dim_v_part], dtype)
                gate = T.alloc_shared([CHUNK_TOKENS, dim_k_part], "float32")
                # The partition's last chunk needs its own tiles: the pipelined loop
                # multi-buffers the ones it reads, and one buffer cannot carry both layouts.
                tail_keys = T.alloc_shared([CHUNK_TOKENS, dim_k_part], dtype)
                tail_values = T.alloc_shared([CHUNK_TOKENS, dim_v_part], dtype)
                tail_gate = T.alloc_shared([CHUNK_TOKENS, dim_k_part], "float32")
                decayed = T.alloc_fragment([CHUNK_TOKENS, dim_k_part], dtype)
                last = T.alloc_fragment([dim_k_part], "float32")
                chunks = T.alloc_local([1], "int32")

                part_tiling.cumsum_offsets(cu_seqlens, part_cum)
                if part < part_cum[num_seqs]:
                    part_tiling.decode(part, part_cum, lo, hi, seq, first)
                    start = T.cast(cu_seqlens[seq[0]], "int32") + first[0]
                    end = T.cast(cu_seqlens[seq[0] + 1], "int32")
                    # The global index of this partition's first chunk, counted here rather
                    # than from a second prefix array the block would also hold.
                    chunks[0] = first[0] // CHUNK_TOKENS
                    for g in T.serial(num_seqs):
                        if g < seq[0]:
                            size = T.cast(cu_seqlens[g + 1] - cu_seqlens[g], "int32")
                            chunks[0] += (size + CHUNK_TOKENS - 1) // CHUNK_TOKENS
                    base = chunks[0]

                    T.fill(state, 0.0)
                    T.fill(reach, 0.0)

                    # Only the chunks lying whole inside both the partition and the sequence
                    # enter the pipelined loop, so its body holds no branch. The trip count
                    # comes from the offsets: a shared-memory read cannot stand in a loop
                    # extent.
                    whole = T.min(end - start, partition_chunks * CHUNK_TOKENS) // CHUNK_TOKENS
                    for i_c in T.Pipelined(whole, num_stages=num_stages):
                        token = start + i_c * CHUNK_TOKENS
                        for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                            chunk_state[base + i_c, i_h, k_offset + i_k, v_offset + i_v] = T.cast(
                                state[i_k, i_v], dtype
                            )
                        if leads:
                            for i_k in T.Parallel(dim_k_part):
                                chunk_reach[base + i_c, i_h, k_offset + i_k] = reach[i_k]
                        T.copy(
                            k[
                                0,
                                token : token + CHUNK_TOKENS,
                                i_h,
                                k_offset : k_offset + dim_k_part,
                            ],
                            keys,
                            disable_tma=True,
                        )
                        T.copy(
                            v[
                                0,
                                token : token + CHUNK_TOKENS,
                                i_h,
                                v_offset : v_offset + dim_v_part,
                            ],
                            values,
                            disable_tma=True,
                        )
                        T.copy(
                            g_cumsum[
                                0,
                                token : token + CHUNK_TOKENS,
                                i_h,
                                k_offset : k_offset + dim_k_part,
                            ],
                            gate,
                            disable_tma=True,
                        )
                        for d in T.Parallel(dim_k_part):
                            last[d] = gate[CHUNK_TOKENS - 1, d]
                        for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                            state[i_k, i_v] = state[i_k, i_v] * T.exp2(last[i_k] * LOG2E)
                        for d in T.Parallel(dim_k_part):
                            reach[d] = reach[d] + last[d]
                        for i, d in T.Parallel(CHUNK_TOKENS, dim_k_part):
                            decayed[i, d] = T.cast(
                                T.cast(keys[i, d], "float32")
                                * T.exp2((last[d] - gate[i, d]) * LOG2E),
                                dtype,
                            )
                        T.gemm(
                            decayed,
                            values,
                            state,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                    # The sequence's last chunk, which falls in this partition only while the
                    # partition has chunks left. A token past the end reads a zero key and
                    # value and repeats the last accumulated gate.
                    if whole < partition_chunks and start + whole * CHUNK_TOKENS < end:
                        token = start + whole * CHUNK_TOKENS
                        for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                            chunk_state[base + whole, i_h, k_offset + i_k, v_offset + i_v] = T.cast(
                                state[i_k, i_v], dtype
                            )
                        if leads:
                            for i_k in T.Parallel(dim_k_part):
                                chunk_reach[base + whole, i_h, k_offset + i_k] = reach[i_k]
                        for i, d in T.Parallel(CHUNK_TOKENS, dim_k_part):
                            tail_gate[i, d] = g_cumsum[
                                0, T.min(token + i, end - 1), i_h, k_offset + d
                            ]
                            tail_keys[i, d] = T.if_then_else(
                                token + i < end,
                                k[0, T.min(token + i, end - 1), i_h, k_offset + d],
                                T.cast(0, dtype),
                            )
                        for i, d in T.Parallel(CHUNK_TOKENS, dim_v_part):
                            tail_values[i, d] = T.if_then_else(
                                token + i < end,
                                v[0, T.min(token + i, end - 1), i_h, v_offset + d],
                                T.cast(0, dtype),
                            )
                        for d in T.Parallel(dim_k_part):
                            last[d] = tail_gate[CHUNK_TOKENS - 1, d]
                        for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                            state[i_k, i_v] = state[i_k, i_v] * T.exp2(last[i_k] * LOG2E)
                        for d in T.Parallel(dim_k_part):
                            reach[d] = reach[d] + last[d]
                        for i, d in T.Parallel(CHUNK_TOKENS, dim_k_part):
                            decayed[i, d] = T.cast(
                                T.cast(tail_keys[i, d], "float32")
                                * T.exp2((last[d] - tail_gate[i, d]) * LOG2E),
                                dtype,
                            )
                        T.gemm(
                            decayed,
                            tail_values,
                            state,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        summary_state[part, i_h, k_offset + i_k, v_offset + i_v] = state[i_k, i_v]
                    if leads:
                        for i_k in T.Parallel(dim_k_part):
                            summary_reach[part, i_h, k_offset + i_k] = reach[i_k]

        return _main

    return _fn


@functools.lru_cache(maxsize=32)
def gla_varlen_scan_kernel(
    total_tokens: int,
    num_seqs: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    partition_chunks: int,
    block_k: int,
    block_v: int,
    num_stages: int,
):
    """Compose one sequence's partition summaries into the state each partition starts from.

    A partition maps its incoming state to ``h * exp(reach) + summary``, and that map is
    associative, so the sequence's walk over its partitions is all that remains of the chain
    the local pass broke. It runs on one block per value tile, over as many partitions as the
    sequence has, which is the chunk count divided by the partition length. The state tile
    is split both ways, because a call with few sequences and few heads has nothing else to
    fill the device with.
    """
    part_tiling = GroupTiling(num_seqs, CHUNK_TOKENS * partition_chunks)
    num_partitions = part_tiling.tile_upper_bound(total_tokens)
    partition_tokens = CHUNK_TOKENS * partition_chunks
    num_tiles = dim_k // block_k * (dim_v // block_v)
    num_v_tiles = dim_v // block_v

    @tilelang.jit(out_idx=[-2, -1], pass_configs=_PASS_CONFIGS)
    def _fn(threads: int = 128):
        @T.prim_func
        def _main(
            summary_state: T.Tensor([num_partitions, heads, dim_k, dim_v], "float32"),
            summary_reach: T.Tensor([num_partitions, heads, dim_k], "float32"),
            initial_state: T.Tensor([num_seqs, heads, dim_k, dim_v], "float32"),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            start_state: T.Tensor([num_partitions, heads, dim_k, dim_v], "float32"),
            final_state: T.Tensor([num_seqs, heads, dim_k, dim_v], "float32"),
        ):
            with T.Kernel(num_seqs * num_tiles, heads, threads=threads) as (bx, i_h):
                seq = bx // num_tiles
                k_offset = bx % num_tiles // num_v_tiles * block_k
                v_offset = bx % num_tiles % num_v_tiles * block_v

                part_cum = T.alloc_shared([num_seqs + 1], "int32")
                state = T.alloc_shared([block_k, block_v], "float32")
                summary = T.alloc_shared([block_k, block_v], "float32")
                reach = T.alloc_shared([block_k], "float32")

                part_tiling.cumsum_offsets(cu_seqlens, part_cum)
                start = T.cast(cu_seqlens[seq], "int32")
                end = T.cast(cu_seqlens[seq + 1], "int32")
                base = part_cum[seq]
                # The trip count comes from the offsets: a shared-memory read cannot stand
                # in a loop extent.
                count = (end - start + partition_tokens - 1) // partition_tokens

                T.copy(
                    initial_state[
                        seq, i_h, k_offset : k_offset + block_k, v_offset : v_offset + block_v
                    ],
                    state,
                )
                # Only the running state carries between partitions, so the summaries ahead
                # load while this one is composed.
                for i_p in T.Pipelined(count, num_stages=num_stages):
                    for i_k, i_v in T.Parallel(block_k, block_v):
                        start_state[base + i_p, i_h, k_offset + i_k, v_offset + i_v] = state[
                            i_k, i_v
                        ]
                    T.copy(
                        summary_state[
                            base + i_p,
                            i_h,
                            k_offset : k_offset + block_k,
                            v_offset : v_offset + block_v,
                        ],
                        summary,
                        disable_tma=True,
                    )
                    T.copy(
                        summary_reach[base + i_p, i_h, k_offset : k_offset + block_k],
                        reach,
                        disable_tma=True,
                    )
                    for i_k, i_v in T.Parallel(block_k, block_v):
                        state[i_k, i_v] = (
                            state[i_k, i_v] * T.exp2(reach[i_k] * LOG2E) + summary[i_k, i_v]
                        )
                for i_k, i_v in T.Parallel(block_k, block_v):
                    final_state[seq, i_h, k_offset + i_k, v_offset + i_v] = state[i_k, i_v]

        return _main

    return _fn


@functools.lru_cache(maxsize=32)
def gla_varlen_partitioned_output_kernel(
    total_tokens: int,
    num_seqs: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    scale: float,
    dtype: str,
    partition_chunks: int,
):
    """Restore the chunk's state, and read it and the causal product into the contractions.

    The state walk leaves a chunk's state partition-local, so this pass completes it: the
    state its partition starts from, decayed by the gate the partition has reached at this
    chunk, plus the local state. That is one more tile read per chunk, and the partitions a
    chunk shares it with read the same one.
    """
    part_tiling = GroupTiling(num_seqs, CHUNK_TOKENS * partition_chunks)
    tiling = GroupTiling(num_seqs, CHUNK_TOKENS)
    num_partitions = part_tiling.tile_upper_bound(total_tokens)
    num_chunks = tiling.tile_upper_bound(total_tokens)
    partition_tokens = CHUNK_TOKENS * partition_chunks

    @tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
    def _fn(threads: int = 128):
        @T.prim_func
        def _main(
            q: T.Tensor([1, total_tokens, heads, dim_k], dtype),
            v: T.Tensor([1, total_tokens, heads, dim_v], dtype),
            g_cumsum: T.Tensor([1, total_tokens, heads, dim_k], "float32"),
            chunk_state: T.Tensor([num_chunks, heads, dim_k, dim_v], dtype),
            chunk_reach: T.Tensor([num_chunks, heads, dim_k], "float32"),
            start_state: T.Tensor([num_partitions, heads, dim_k, dim_v], "float32"),
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
                values = T.alloc_shared([CHUNK_TOKENS, dim_v], dtype)
                weights = T.alloc_shared([CHUNK_TOKENS, CHUNK_TOKENS], dtype)
                q_gated = T.alloc_shared([CHUNK_TOKENS, dim_k], dtype)
                state = T.alloc_shared([dim_k, dim_v], dtype)
                acc = T.alloc_fragment([CHUNK_TOKENS, dim_v], "float32")
                partitions = T.alloc_local([1], "int32")

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                if chunk < tile_cum[num_seqs]:
                    tiling.decode(chunk, tile_cum, lo, hi, seq, first)
                    start = T.cast(cu_seqlens[seq[0]], "int32") + first[0]
                    end = T.cast(cu_seqlens[seq[0] + 1], "int32")
                    # The partition this chunk lies in, counted here rather than from a second
                    # prefix array the block would also hold.
                    partitions[0] = first[0] // partition_tokens
                    for g in T.serial(num_seqs):
                        if g < seq[0]:
                            size = T.cast(cu_seqlens[g + 1] - cu_seqlens[g], "int32")
                            partitions[0] += (size + partition_tokens - 1) // partition_tokens
                    part = partitions[0]

                    # The query and its gate are each read once, so neither is staged: the
                    # two tiles they would occupy bound this block's shared footprint.
                    for i, d in T.Parallel(CHUNK_TOKENS, dim_k):
                        q_gated[i, d] = T.cast(
                            T.cast(q[0, T.min(start + i, end - 1), i_h, d], "float32")
                            * T.exp2(g_cumsum[0, T.min(start + i, end - 1), i_h, d] * LOG2E),
                            dtype,
                        )
                    for i, d in T.Parallel(CHUNK_TOKENS, dim_v):
                        values[i, d] = v[0, T.min(start + i, end - 1), i_h, d]
                    # The causal pass launches only the sub-block pairs at or below the
                    # diagonal, so the product above the diagonal is read as the zero it is.
                    for i, j in T.Parallel(CHUNK_TOKENS, CHUNK_TOKENS):
                        weights[i, j] = T.if_then_else(
                            j <= i, causal[0, T.min(start + i, end - 1), i_h, j], T.cast(0, dtype)
                        )
                    for d, j in T.Parallel(dim_k, dim_v):
                        state[d, j] = T.cast(
                            start_state[part, i_h, d, j]
                            * T.exp2(chunk_reach[chunk, i_h, d] * LOG2E)
                            + T.cast(chunk_state[chunk, i_h, d, j], "float32"),
                            dtype,
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


class GLAVarlenPrefillPartitionedFwdKernel(Kernel, GLAInferenceFwdInterface):
    """Packed prefill whose state walk is partitioned, for the calls where that pays.

    Each chunk-parallel launch covers ``total_tokens // 64 + num_sequences`` chunks, the most
    the offsets can describe, and a block past the count the offsets give retires at once.
    The state walk is laid out the same way over partitions of ``partition_chunks`` chunks.
    """

    supported_archs = [90]
    preferred_over = frozenset({"gla_varlen_prefill"})

    # Threads per block. The causal product's GEMM tiles a 16-wide operand, which more than
    # two warps cannot split into whole tiles; the walk's key extent bounds its block the same
    # way. Re-fit by timing the packed manifest rows at 64, 128 and 256.
    _causal_threads = 64
    _state_threads = 128
    _wide_threads = 256

    # Chunks the walk inside a partition prefetches ahead. Re-fit with the packed manifest
    # rows over 2 to 6; a stage costs shared memory the state tile also needs.
    _state_stages = 2

    # Key and value channels of the state tile one block of the walk holds. A key slice reads
    # only its own key channels and its own slice of the float32 gate; a value slice reads the
    # keys and the gate again, which is why the value extent stays as wide as the head. Re-fit
    # by timing the packed manifest rows at 16, 32 and 64 against 64 and 128.
    _state_tile_k = 32
    _state_tile_v = 128

    # State tile one block of the scan holds, and the partitions it prefetches. The tile is
    # what gives a call with few sequences and heads enough blocks. Re-fit by timing the
    # packed manifest rows.
    _scan_tile_k = 32
    _scan_tile_v = 32
    _scan_stages = 2

    # Partition lengths the walk may take, longest first. A longer partition carries fewer
    # summaries and so less traffic; a shorter one gives the device more independent blocks.
    _partition_lengths = (16, 8, 4, 2)

    # Blocks per multiprocessor a walk is sized to launch. Re-fit by timing the packed
    # manifest rows over the lengths above.
    _blocks_per_sm = 1

    @classmethod
    def refusal(cls, call: GLAInferenceCallSpec) -> Optional[str]:
        return head_count_refusal(call.heads) or super().refusal(call)

    @classmethod
    def applies(cls, call: GLAInferenceCallSpec) -> bool:
        """A packed or part-chunk call whose sequences the per-sequence walk serves badly.

        Two cases reach that, and each is read from the shapes alone. A call whose rows are
        equal states its own chunk count, so a row holding more chunks than the longest
        partition has something to partition. A packed call does not, so what is read
        instead is the blocks a per-sequence walk would launch: past what the device holds
        resident, the state slicing it takes is re-read traffic rather than parallelism.
        """
        if not serves_extents(call):
            return False
        if not call.varlen:
            return (
                call.seq_len > 1
                and call.seq_len % CHUNK_TOKENS != 0
                and call.seq_len // CHUNK_TOKENS > cls._partition_lengths[0]
            )
        # The per-sequence walk splits the state tile to the GEMM's minimum operand extent
        # along the key axis and once along the value axis, which is the finest split it has.
        per_sequence_blocks = call.num_sequences * call.heads * (call.dim_k // GEMM_MIN_N) * 2
        return per_sequence_blocks > cls._blocks_per_sm * call.sm_count

    @classmethod
    def entry_for(cls, call: GLAInferenceCallSpec) -> Entry:
        k_partitions = max(1, call.dim_k // cls._state_tile_k)
        v_partitions = max(1, call.dim_v // cls._state_tile_v)
        # The longest partition whose blocks still cover the device.
        chunks = call.batch * call.seq_len // CHUNK_TOKENS
        slices = k_partitions * v_partitions
        target = cls._blocks_per_sm * call.sm_count
        partition_chunks = cls._partition_lengths[-1]
        for length in cls._partition_lengths:
            if chunks // length * slices * call.heads >= target:
                partition_chunks = length
                break
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
            partition_chunks=partition_chunks,
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
        partition_chunks: int,
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
        self._local = gla_varlen_local_state_kernel(
            total,
            num_sequences,
            heads,
            dim_k,
            dim_v,
            name,
            k_partitions,
            v_partitions,
            partition_chunks,
            self._state_stages,
        )(self._state_threads)
        self._scan = gla_varlen_scan_kernel(
            total,
            num_sequences,
            heads,
            dim_k,
            dim_v,
            partition_chunks,
            self._scan_tile_k,
            self._scan_tile_v,
            self._scan_stages,
        )(self._state_threads)
        self._causal = gla_varlen_causal_kernel(total, num_sequences, heads, dim_k, scale, name)(
            self._causal_threads
        )
        self._output = gla_varlen_partitioned_output_kernel(
            total, num_sequences, heads, dim_k, dim_v, scale, name, partition_chunks
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
        chunk_state, chunk_reach, summary_state, summary_reach = self._local(
            k.view(packed), v.view(packed), gate, cu_seqlens
        )
        start_state, final_state = self._scan(summary_state, summary_reach, state, cu_seqlens)
        causal = self._causal(q.view(packed), k.view(packed), gate, cu_seqlens)
        o = self._output(
            q.view(packed),
            v.view(packed),
            gate,
            chunk_state,
            chunk_reach,
            start_state,
            causal,
            cu_seqlens,
        )
        return o.view(self.batch, self.seq_len, self.heads, self.dim_v), final_state
