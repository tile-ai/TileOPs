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

        @T.macro
        def compute_value_major(q, k, v, g, beta, h0, o, ht, a_log, dt_bias, slots):
            with T.Kernel(num_seqs * HV * tiles, threads=threads) as (block,):
                tx = T.get_thread_binding()
                iv = block % tiles
                iseq = block // (tiles * HV)
                ihv = block // tiles - iseq * HV
                ih = ihv // group
                row = iv * BV + tx // lane_group
                rank = tx % lane_group
                q_s = T.alloc_shared([K], accum)
                k_s = T.alloc_shared([K], accum)
                d_s = T.alloc_shared([K], accum)
                state = T.alloc_local([key_chunk], accum)
                old = T.alloc_var(accum, init=0.0)
                out = T.alloc_var(accum, init=0.0)
                q_square = T.alloc_var(accum, init=0.0)
                k_square = T.alloc_var(accum, init=0.0)

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
                for i in T.serial(T.ceildiv(K, threads)):
                    j = tx + i * threads
                    if j < K:
                        q_s[j] = T.cast(q[0, iseq, ih, j], accum)
                        k_s[j] = T.cast(k[0, iseq, ih, j], accum)
                        d_s[j] = channel_decay(g, a_log, dt_bias, iseq, ihv, j)
                T.sync_threads()
                q_gain, k_gain = scale, 1.0
                if l2norm:
                    # Every warp folds the whole row, so both norms land in every thread with
                    # no second pass over shared memory.
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

        @T.macro
        def compute(q, k, v, g, beta, h0, o, ht, a_log, dt_bias, slots):
            with T.Kernel(num_seqs, HV, tiles, threads=threads) as (iseq, ihv, iv):
                ih = ihv // group
                q_s = T.alloc_shared([K], accum)
                k_s = T.alloc_shared([K], accum)
                d_s = T.alloc_shared([K], accum)
                qrow = T.alloc_fragment([1, K], accum)
                krow = T.alloc_fragment([1, K], accum)
                qsum = T.alloc_fragment([1], accum)
                ksum = T.alloc_fragment([1], accum)
                state = T.alloc_fragment([K, BV], accum)
                part = T.alloc_fragment([K, BV], accum)
                old = T.alloc_fragment([BV], accum)
                out = T.alloc_fragment([BV], accum)
                newv = T.alloc_fragment([BV], accum)

                # The two row norms come off a block-wide reduction: a decode step
                # is short enough that one thread summing K values is its tail.
                for _i, j in T.Parallel(1, K):
                    qrow[0, j] = T.cast(q[0, iseq, ih, j], accum)
                    krow[0, j] = T.cast(k[0, iseq, ih, j], accum)
                for _i, j in T.Parallel(1, K):
                    qrow[0, j] = qrow[0, j] * qrow[0, j]
                    krow[0, j] = krow[0, j] * krow[0, j]
                T.reduce_sum(qrow, qsum, dim=1)
                T.reduce_sum(krow, ksum, dim=1)
                for _i, j in T.Parallel(1, K):
                    q_s[j] = (
                        T.cast(q[0, iseq, ih, j], accum)
                        * (T.rsqrt(qsum[0] + L2NORM_EPS) if l2norm else 1.0)
                        * scale
                    )
                    k_s[j] = T.cast(k[0, iseq, ih, j], accum) * (
                        T.rsqrt(ksum[0] + L2NORM_EPS) if l2norm else 1.0
                    )
                    d_s[j] = channel_decay(g, a_log, dt_bias, iseq, ihv, j)
                T.sync_threads()

                # The decayed state, held for the whole step.
                for i, j in T.Parallel(K, BV):
                    state[i, j] = d_s[i] * h0[state_row(slots, iseq), ihv, i, iv * BV + j]
                    part[i, j] = k_s[i] * state[i, j]
                T.reduce_sum(part, old, dim=0)
                for j in T.Parallel(BV):
                    newv[j] = T.cast(beta[0, iseq, ihv], accum) * (
                        T.cast(v[0, iseq, ihv, iv * BV + j], accum) - old[j]
                    )
                for i, j in T.Parallel(K, BV):
                    state[i, j] = state[i, j] + k_s[i] * newv[j]
                    part[i, j] = q_s[i] * state[i, j]
                    ht[state_row(slots, iseq), ihv, i, iv * BV + j] = state[i, j]
                T.reduce_sum(part, out, dim=0)
                for j in T.Parallel(BV):
                    o[0, iseq, ihv, iv * BV + j] = T.cast(out[j], dtype)

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
