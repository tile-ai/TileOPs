"""The TileLang program for a single Kimi Delta Attention (KDA) decode step.

A decode step reads the recurrent state once and writes it once, so what
decides its latency is how much of the machine is reading. One block takes one
(sequence, value head) and a slice of the value axis, holds that slice of the
state in registers for the whole step, and never reads it twice: the decayed
state, the delta correction, the new state and the output all come off one
pass. Slicing the value axis is what gives a batch of one enough blocks.
"""

import functools

import tilelang
import tilelang.language as T

from tileops._kernels.constants import LOG2E
from tileops._kernels.linear_attention.kda.gate import log_decay

__all__ = ["decode_program"]


@functools.lru_cache(maxsize=32)
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
):
    """Build the decode program for one static packed shape."""
    H, HV, K, V = heads, value_heads, dim_k, dim_v
    # Added under the square root of each row norm before taking its reciprocal.
    l2_eps = 1e-6
    BV = min(value_tile, V)
    tiles = V // BV
    group = HV // H
    accum = "float32"

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.macro
        def compute(q, k, v, g, beta, h0, o, ht, a_log, dt_bias):
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
                        * (T.rsqrt(qsum[0] + l2_eps) if l2norm else 1.0)
                        * scale
                    )
                    k_s[j] = T.cast(k[0, iseq, ih, j], accum) * (
                        T.rsqrt(ksum[0] + l2_eps) if l2norm else 1.0
                    )
                    if gate_in_kernel:
                        d_s[j] = T.exp2(
                            log_decay(
                                g[0, iseq, ihv, j],
                                a_log[ihv],
                                dt_bias[ihv * K + j] if has_dt_bias else T.float32(0.0),
                                lower_bound,
                            )
                            * LOG2E
                        )
                    else:
                        d_s[j] = T.exp2(T.cast(g[0, iseq, ihv, j], accum) * LOG2E)
                T.sync_threads()

                # The decayed state, held for the whole step.
                for i, j in T.Parallel(K, BV):
                    state[i, j] = d_s[i] * h0[iseq, ihv, i, iv * BV + j]
                    part[i, j] = k_s[i] * state[i, j]
                T.reduce_sum(part, old, dim=0)
                for j in T.Parallel(BV):
                    newv[j] = T.cast(beta[0, iseq, ihv], accum) * (
                        T.cast(v[0, iseq, ihv, iv * BV + j], accum) - old[j]
                    )
                for i, j in T.Parallel(K, BV):
                    state[i, j] = state[i, j] + k_s[i] * newv[j]
                    part[i, j] = q_s[i] * state[i, j]
                    ht[iseq, ihv, i, iv * BV + j] = state[i, j]
                T.reduce_sum(part, out, dim=0)
                for j in T.Parallel(BV):
                    o[0, iseq, ihv, iv * BV + j] = T.cast(out[j], dtype)

        if gate_in_kernel:

            @T.prim_func
            def main(
                q: T.Tensor([1, num_seqs, H, K], dtype),
                k: T.Tensor([1, num_seqs, H, K], dtype),
                v: T.Tensor([1, num_seqs, HV, V], dtype),
                g: T.Tensor([1, num_seqs, HV, K], dtype),
                beta: T.Tensor([1, num_seqs, HV], dtype),
                h0: T.Tensor([num_seqs, HV, K, V], accum),
                a_log: T.Tensor([HV], "float32"),
                dt_bias: T.Tensor([HV * K if has_dt_bias else HV], "float32"),
                o: T.Tensor([1, num_seqs, HV, V], dtype),
                ht: T.Tensor([num_seqs, HV, K, V], accum),
            ):
                compute(q, k, v, g, beta, h0, o, ht, a_log, dt_bias)
        else:

            @T.prim_func
            def main(
                q: T.Tensor([1, num_seqs, H, K], dtype),
                k: T.Tensor([1, num_seqs, H, K], dtype),
                v: T.Tensor([1, num_seqs, HV, V], dtype),
                g: T.Tensor([1, num_seqs, HV, K], dtype),
                beta: T.Tensor([1, num_seqs, HV], dtype),
                h0: T.Tensor([num_seqs, HV, K, V], accum),
                o: T.Tensor([1, num_seqs, HV, V], dtype),
                ht: T.Tensor([num_seqs, HV, K, V], accum),
            ):
                compute(q, k, v, g, beta, h0, o, ht, None, None)

        return main

    return build()
