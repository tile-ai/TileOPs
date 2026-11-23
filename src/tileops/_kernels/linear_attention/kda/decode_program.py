"""The TileLang program for a single Kimi Delta Attention (KDA) decode step.

One block takes one (sequence, value head) and a slice of the value axis, and holds that
slice of the state in registers for the whole step: the decayed state, the delta correction,
the new state and the output all come off one read of it.

A value-major state keeps a value row's key positions contiguous, so its block
takes whole rows instead: a lane group splits one row's keys in 16-byte vectors,
and the rows of a block are the value slice.
"""

import math

import tilelang
import tilelang.language as T

from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import LOG2E
from tileops._kernels.linear_attention.delta_decode import key_of
from tileops._kernels.linear_attention.gdn.prefill_common import L2NORM_EPS
from tileops._kernels.linear_attention.kda.gate import log_decay
from tileops._kernels.state_pool import state_row
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["decode_program"]


@target_cache(maxsize=32)
def decode_program(
    heads: int,
    value_heads: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    scale: float,
    l2norm: bool,
    num_seqs: int,
    value_tile: int = 32,
    threads: int = 256,
    *,
    gate_in_kernel: bool = False,
    has_dt_bias: bool = False,
    lower_bound: float | None = None,
    state_v_first: bool = False,
    pooled: bool = False,
):
    """Build the decode program for one static packed shape.

    *value_tile* is the value columns one block owns. *state_v_first* lays the state
    out value-major, ``[N, HV, V, K]``; *pooled* reads sequence ``n``'s state from slot
    ``slots[n]`` of a state pool and writes the advanced state back there, returning
    only the output.
    """
    H, HV, K, V = heads, value_heads, dim_k, dim_v
    BV = min(value_tile, V)
    tiles = V // BV
    group = HV // H
    accum = "float32"
    # Over a value-major state, the lanes that split one value row's keys and the run of
    # contiguous key positions each moves as one vector.
    lane_group = threads // BV
    key_chunk = K // lane_group
    key_run = math.gcd(key_chunk, 4)

    @tilelang.jit(
        out_idx=[-1] if pooled else [-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        def channel_decay(g, a_log, dt_bias, iseq, ihv, j):
            """The decay factor of key channel *j*."""
            if gate_in_kernel:
                return T.exp2(
                    log_decay(
                        g[0, iseq, ihv, j],
                        a_log[ihv],
                        dt_bias[ihv * K + j] if has_dt_bias else T.float32(0.0),
                        lower_bound,
                    )
                    * LOG2E
                )
            return T.exp2(T.cast(g[0, iseq, ihv, j], accum) * LOG2E)

        def coordinates(block):
            """The value tile, sequence, value head and key head of a block launched in the
            state's memory order."""
            iseq = block // (tiles * HV)
            ihv = block // tiles - iseq * HV
            return block % tiles, iseq, ihv, ihv // group

        @T.macro
        def stage_rows(q, k, g, a_log, dt_bias, iseq, ih, ihv, tx, q_s, k_s, d_s):
            """Stage the query, key and decay rows in shared memory."""
            for i in T.serial(T.ceildiv(K, threads)):
                j = tx + i * threads
                if j < K:
                    q_s[j] = T.cast(q[0, iseq, ih, j], accum)
                    k_s[j] = T.cast(k[0, iseq, ih, j], accum)
                    d_s[j] = channel_decay(g, a_log, dt_bias, iseq, ihv, j)
            T.sync_threads()

        @T.macro
        def row_gains(q_s, k_s, tx):
            """The query and key gains: the scale, and under l2norm the inverse row norms."""
            q_gain, k_gain = scale, 1.0
            if l2norm:
                q_square = T.alloc_var(accum, init=0.0)
                k_square = T.alloc_var(accum, init=0.0)
                # Every warp folds the whole row, so both norms land in every thread with no
                # second pass over shared memory.
                lane = tx % WARP_LANES
                for i in T.serial(T.ceildiv(K, WARP_LANES)):
                    j = lane + i * WARP_LANES
                    if j < K:
                        q_square += q_s[j] * q_s[j]
                        k_square += k_s[j] * k_s[j]
                for stage in T.unroll(WARP_SHUFFLE_STAGES):
                    q_square += T.shfl_xor(q_square, 1 << stage, width=WARP_LANES)
                    k_square += T.shfl_xor(k_square, 1 << stage, width=WARP_LANES)
                q_gain = T.rsqrt(q_square + L2NORM_EPS) * scale
                k_gain = T.rsqrt(k_square + L2NORM_EPS)
            return q_gain, k_gain

        @T.macro
        def compute_value_major(q, k, v, g, beta, h0, o, ht, a_log, dt_bias, slots):
            with T.Kernel(num_seqs * HV * tiles, threads=threads) as (block,):
                tx = T.get_thread_binding()
                iv, iseq, ihv, ih = coordinates(block)
                row = iv * BV + tx // lane_group
                rank = tx % lane_group
                q_s = T.alloc_shared([K], accum)
                k_s = T.alloc_shared([K], accum)
                d_s = T.alloc_shared([K], accum)
                state = T.alloc_local([key_chunk], accum)
                old = T.alloc_var(accum, init=0.0)
                out = T.alloc_var(accum, init=0.0)

                # The state loads go out first, so they are in flight while the block forms its
                # query, key and decay rows.
                for run in T.serial(key_chunk // key_run):
                    for e in T.vectorized(key_run):
                        state[run * key_run + e] = h0[
                            state_row(slots, iseq),
                            ihv,
                            row,
                            key_of(rank, run * key_run + e, lane_group, key_run),
                        ]
                stage_rows(q, k, g, a_log, dt_bias, iseq, ih, ihv, tx, q_s, k_s, d_s)
                q_gain, k_gain = row_gains(q_s, k_s, tx)

                for i in T.serial(key_chunk):
                    state[i] = state[i] * d_s[key_of(rank, i, lane_group, key_run)]
                    old += state[i] * k_s[key_of(rank, i, lane_group, key_run)]
                for stage in T.unroll(lane_group.bit_length() - 1):
                    old += T.shfl_xor(old, 1 << stage, width=lane_group)
                written = T.cast(beta[0, iseq, ihv], accum) * (
                    T.cast(v[0, iseq, ihv, row], accum) - old * k_gain
                )
                for i in T.serial(key_chunk):
                    state[i] = (
                        state[i] + k_s[key_of(rank, i, lane_group, key_run)] * k_gain * written
                    )
                    out += state[i] * q_s[key_of(rank, i, lane_group, key_run)]
                for run in T.serial(key_chunk // key_run):
                    for e in T.vectorized(key_run):
                        ht[
                            state_row(slots, iseq),
                            ihv,
                            row,
                            key_of(rank, run * key_run + e, lane_group, key_run),
                        ] = state[run * key_run + e]
                for stage in T.unroll(lane_group.bit_length() - 1):
                    out += T.shfl_xor(out, 1 << stage, width=lane_group)
                if rank == 0:
                    o[0, iseq, ihv, row] = T.cast(out * q_gain, dtype)

        # Over a key-major state, a lane owns a run of value columns that moves as one 16-byte
        # vector of FP32; the lanes that share a column run split its key rows.
        vector = 4
        col_lanes = BV // vector
        row_groups = threads // col_lanes
        rows_per = K // row_groups

        @T.macro
        def share_sums(part, part_s, rank, lane):
            """Publish a lane's partial sums of its column run for the other ranks."""
            for e in T.unroll(vector):
                part_s[rank, lane * vector + e] = part[e]
            T.sync_threads()

        @T.macro
        def sum_ranks(part_s, lane, total):
            """Sum the partial sums of the lane's column run over every rank."""
            for e in T.unroll(vector):
                total[e] = 0.0
            for r in T.unroll(row_groups):
                for e in T.unroll(vector):
                    total[e] += part_s[r, lane * vector + e]

        @T.macro
        def compute(q, k, v, g, beta, h0, o, ht, a_log, dt_bias, slots):
            # The blocks go in the state's memory order, so the blocks resident at once read
            # neighbouring states.
            with T.Kernel(num_seqs * HV * tiles, threads=threads) as (block,):
                tx = T.get_thread_binding()
                iv, iseq, ihv, ih = coordinates(block)
                lane = tx % col_lanes
                rank = tx // col_lanes
                v0 = iv * BV + lane * vector
                q_s = T.alloc_shared([K], accum)
                k_s = T.alloc_shared([K], accum)
                d_s = T.alloc_shared([K], accum)
                # One row of partial sums per lane rank; the second product reuses it.
                part_s = T.alloc_shared([row_groups, BV], accum)
                state = T.alloc_local([rows_per * vector], accum)
                part = T.alloc_local([vector], accum)
                written = T.alloc_local([vector], accum)

                # The state loads go out first, so they are in flight while the block forms its
                # query, key and decay rows.
                for r in T.unroll(rows_per):
                    for e in T.vectorized(vector):
                        state[r * vector + e] = h0[
                            state_row(slots, iseq), ihv, rank + r * row_groups, v0 + e
                        ]
                stage_rows(q, k, g, a_log, dt_bias, iseq, ih, ihv, tx, q_s, k_s, d_s)
                q_gain, k_gain = row_gains(q_s, k_s, tx)

                for e in T.unroll(vector):
                    part[e] = 0.0
                for r in T.unroll(rows_per):
                    for e in T.unroll(vector):
                        state[r * vector + e] = state[r * vector + e] * d_s[rank + r * row_groups]
                        # The gains go into each product, as the normalized rows would: a
                        # state near the FP32 limit stays finite.
                        part[e] += state[r * vector + e] * (k_s[rank + r * row_groups] * k_gain)
                share_sums(part, part_s, rank, lane)
                sum_ranks(part_s, lane, written)
                for e in T.unroll(vector):
                    written[e] = T.cast(beta[0, iseq, ihv], accum) * (
                        T.cast(v[0, iseq, ihv, v0 + e], accum) - written[e]
                    )
                    part[e] = 0.0
                for r in T.unroll(rows_per):
                    for e in T.unroll(vector):
                        state[r * vector + e] = (
                            state[r * vector + e] + k_s[rank + r * row_groups] * k_gain * written[e]
                        )
                        part[e] += state[r * vector + e] * (q_s[rank + r * row_groups] * q_gain)
                for r in T.unroll(rows_per):
                    for e in T.vectorized(vector):
                        ht[state_row(slots, iseq), ihv, rank + r * row_groups, v0 + e] = state[
                            r * vector + e
                        ]
                # Every lane has read the first sums before any lane overwrites them.
                T.sync_threads()
                share_sums(part, part_s, rank, lane)
                if rank == 0:
                    sum_ranks(part_s, lane, written)
                    for e in T.unroll(vector):
                        o[0, iseq, ihv, v0 + e] = T.cast(written[e], dtype)

        body = compute_value_major if state_v_first else compute
        state_shape = [num_seqs, HV, V, K] if state_v_first else [num_seqs, HV, K, V]
        pool_shape = [T.dynamic("pool"), *state_shape[1:]]
        # A gate parameter a build without the raw gate leaves unread is declared one element
        # wide.
        a_log_shape = [HV] if gate_in_kernel else [1]
        dt_bias_shape = [HV * K if has_dt_bias else HV] if gate_in_kernel else [1]
        if pooled:

            @T.prim_func
            def main(
                q: T.Tensor([1, num_seqs, H, K], dtype),
                k: T.Tensor([1, num_seqs, H, K], dtype),
                v: T.Tensor([1, num_seqs, HV, V], dtype),
                g: T.Tensor([1, num_seqs, HV, K], dtype),
                beta: T.Tensor([1, num_seqs, HV], dtype),
                h0: T.Tensor(pool_shape, accum),
                slots: T.Tensor([num_seqs], "int32"),
                a_log: T.Tensor(a_log_shape, "float32"),
                dt_bias: T.Tensor(dt_bias_shape, "float32"),
                o: T.Tensor([1, num_seqs, HV, V], dtype),
            ):
                body(q, k, v, g, beta, h0, o, h0, a_log, dt_bias, slots)
        else:

            @T.prim_func
            def main(
                q: T.Tensor([1, num_seqs, H, K], dtype),
                k: T.Tensor([1, num_seqs, H, K], dtype),
                v: T.Tensor([1, num_seqs, HV, V], dtype),
                g: T.Tensor([1, num_seqs, HV, K], dtype),
                beta: T.Tensor([1, num_seqs, HV], dtype),
                h0: T.Tensor(state_shape, accum),
                a_log: T.Tensor(a_log_shape, "float32"),
                dt_bias: T.Tensor(dt_bias_shape, "float32"),
                o: T.Tensor([1, num_seqs, HV, V], dtype),
                ht: T.Tensor(state_shape, accum),
            ):
                body(q, k, v, g, beta, h0, o, ht, a_log, dt_bias, None)

        return main

    return build()
