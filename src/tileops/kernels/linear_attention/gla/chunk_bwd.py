"""GLA (Gated Linear Attention) backward kernel — TileLang implementation.

Three launches:
  1. Every chunk in parallel: its contribution to the state gradient,
     W = scale * (q exp(g))^T do, and its gate's last cumsum row.
  2. A walk of the state gradient from the last chunk to the first, D <- exp(g_last) D + W,
     with no matmul on its serial chain.
  3. Every chunk in parallel: dq, dk, dv and dg from the chunk state the forward handed it and
     the state gradient the walk handed it.

h is read from the forward pass's h_out (no recomputation).

Reference:
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py
"""

import functools
from typing import Callable, Optional, Tuple

import tilelang
import torch
from tilelang import language as T
from tilelang.profiler import do_bench

from tileops.kernels.constants import BF16_SPLIT_EXP2_SPAN, LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    GLAChunkBwdInterface,
    GLAChunkCall,
    head_count_refusal,
)
from tileops.kernels.linear_attention.gla.chunk_fwd import (
    chunk_gate_cumsum,
    gla_state_walk_kernel,
    state_walk_plan,
)
from tileops.utils import get_sm_version

__all__ = ["GLAChunkBwdKernel"]


@functools.lru_cache(maxsize=32)
def _gla_bwd_update_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
) -> Callable:
    """Each chunk's contribution to the state gradient, every chunk in parallel.

    One block a (batch, head, chunk) takes the gate cumsum g, writes its last row g_last, and
    writes W = scale * (q * exp(g))^T do, so that the gradient of the state the chunk is handed
    is exp(g_last) * D + W for the gradient D of the state it leaves.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _update_func(threads=128):
        @T.prim_func
        def _main(
            q: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            do: T.Tensor([batch, seq_len, heads, dim_v], dtype),
            g: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            update: T.Tensor([batch, num_chunks, heads, dim_k, dim_v], accum_dtype),
            g_last: T.Tensor([batch, num_chunks, heads, dim_k], accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                start = i_c * chunk_size

                g_s = T.alloc_shared([chunk_size, dim_k], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                do_s = T.alloc_shared([chunk_size, dim_v], dtype)
                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)

                T.copy(g[i_b, start : start + chunk_size, i_h, :], g_s, disable_tma=True)
                T.copy(do[i_b, start : start + chunk_size, i_h, :], do_s, disable_tma=True)
                chunk_gate_cumsum(g_s, g_cumsum_s, chunk_size, dim_k, threads, dtype)

                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    q_gated_s[i_t, i_k] = T.cast(
                        T.cast(q[i_b, start + i_t, i_h, i_k], accum_dtype)
                        * T.exp2(g_cumsum_s[i_t, i_k] * LOG2E),
                        dtype,
                    )
                acc = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                T.clear(acc)
                T.gemm(q_gated_s, do_s, acc, transpose_A=True)
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    update[i_b, i_c, i_h, i_k, i_v] = acc[i_k, i_v] * scale
                for i_k in T.Parallel(dim_k):
                    g_last[i_b, i_c, i_h, i_k] = g_cumsum_s[chunk_size - 1, i_k]

        return _main

    return _update_func


@T.macro
def _split_into_ones_product(part, s0, s1, s2, ones_s, acc, split_dtype: str):
    """acc += ones_s @ part, *part* (float32) split into three bfloat16 parts s0 + s1 + s2.

    The ones are exact in bfloat16 and the three parts carry float32's precision, so the
    product sums *part* along the rows exactly to float32. *part* is left holding the
    remainder after its leading part.
    """
    rows, cols = part.shape
    for i_t, i_k in T.Parallel(rows, cols):
        s0[i_t, i_k] = T.cast(part[i_t, i_k], split_dtype)
    for i_t, i_k in T.Parallel(rows, cols):
        part[i_t, i_k] = part[i_t, i_k] - T.cast(s0[i_t, i_k], "float32")
    for i_t, i_k in T.Parallel(rows, cols):
        s1[i_t, i_k] = T.cast(part[i_t, i_k], split_dtype)
    for i_t, i_k in T.Parallel(rows, cols):
        s2[i_t, i_k] = T.cast(part[i_t, i_k] - T.cast(s1[i_t, i_k], "float32"), split_dtype)
    T.gemm(ones_s, s0, acc)
    T.gemm(ones_s, s1, acc)
    T.gemm(ones_s, s2, acc)


@functools.lru_cache(maxsize=32)
def _gla_bwd_chunk_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
) -> Callable:
    """dq, dk, dv and dg of every chunk in parallel, one block a (batch, head, chunk).

    With G the chunk's gate cumsum, S the state it is handed, D the gradient of the state it
    leaves and M the causal mask, A = scale M (q exp(G)) (k exp(-G))^T and dA = scale M do v^T:

    - dv = A^T do + (k exp(G_last - G)) D;
    - dq = exp(G) (dA (k exp(-G)) + scale do S^T);
    - dk = exp(-G) (dA^T (q exp(G))) + exp(G_last - G) (v D^T);
    - dg_t = sum_{t' >= t} (q dq - k dk_intra) + sum_{t' < t} k dk_state, the last row also
      taking exp(G_last) sum_v S D, where dk_state is dk's term in D and dk_intra the rest.

    Products with q exp(G) and k exp(-G) run on tensor cores; a chunk whose decay leaves
    those factors unrepresentable takes the exact per-pair form. A 16-bit input holds each
    factor and dA as a bfloat16 pair, which keeps float32-level precision.
    """
    accum_dtype = "float32"
    # Every product into an accumulator must run on one instruction type.
    paired = dtype != "float32"
    factor_dtype = "bfloat16" if paired else accum_dtype
    split_dtype = "bfloat16"
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _chunk_func(threads=128):
        @T.prim_func
        def _main(
            q: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            k: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            v: T.Tensor([batch, seq_len, heads, dim_v], dtype),
            g: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            do: T.Tensor([batch, seq_len, heads, dim_v], dtype),
            h: T.Tensor([batch, num_chunks + 1, heads, dim_k, dim_v], accum_dtype),
            dh: T.Tensor([batch, num_chunks + 1, heads, dim_k, dim_v], accum_dtype),
            dq: T.Tensor([batch, seq_len, heads, dim_k], accum_dtype),
            dk: T.Tensor([batch, seq_len, heads, dim_k], accum_dtype),
            dv: T.Tensor([batch, seq_len, heads, dim_v], accum_dtype),
            dg: T.Tensor([batch, seq_len, heads, dim_k], accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                start = i_c * chunk_size

                g_s = T.alloc_shared([chunk_size, dim_k], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                do_s = T.alloc_shared([chunk_size, dim_v], dtype)
                dh_s = T.alloc_shared([dim_k, dim_v], dtype)
                da_hi_s = T.alloc_shared([chunk_size, chunk_size], factor_dtype)
                da_lo_s = T.alloc_shared([chunk_size, chunk_size], factor_dtype)
                k_decay_s = T.alloc_shared([chunk_size, dim_k], dtype)
                q_hi_s = T.alloc_shared([chunk_size, dim_k], factor_dtype)
                q_lo_s = T.alloc_shared([chunk_size, dim_k], factor_dtype)
                k_hi_s = T.alloc_shared([chunk_size, dim_k], factor_dtype)
                k_lo_s = T.alloc_shared([chunk_size, dim_k], factor_dtype)
                a_s = T.alloc_shared([chunk_size, chunk_size], dtype)
                h_s = T.alloc_shared([dim_k, dim_v], dtype)
                tail_s = T.alloc_shared([dim_k], accum_dtype)
                ones_s = T.alloc_shared([chunk_size, chunk_size], split_dtype)
                dg0_s = T.alloc_shared([chunk_size, dim_k], split_dtype)
                dg1_s = T.alloc_shared([chunk_size, dim_k], split_dtype)
                dg2_s = T.alloc_shared([chunk_size, dim_k], split_dtype)
                spans = T.alloc_fragment([dim_k], accum_dtype)
                span = T.alloc_fragment([1], accum_dtype)

                T.copy(g[i_b, start : start + chunk_size, i_h, :], g_s, disable_tma=True)
                chunk_gate_cumsum(g_s, g_cumsum_s, chunk_size, dim_k, threads, dtype)

                # The chunk's whole decay sets the range of exp(G) and exp(-G).
                for i_k in T.Parallel(dim_k):
                    spans[i_k] = -g_cumsum_s[chunk_size - 1, i_k] * LOG2E
                T.reduce_max(spans, span, dim=0)

                T.copy(v[i_b, start : start + chunk_size, i_h, :], v_s, disable_tma=True)
                T.copy(do[i_b, start : start + chunk_size, i_h, :], do_s, disable_tma=True)

                # exp(G_last) sum_v S D, 32 state columns at a time: a wider fragment has no
                # register layout at some (dim_k, dim_v).
                state_cols = 32
                state_dot = T.alloc_fragment([dim_k, state_cols], accum_dtype)
                state_dot_sum = T.alloc_fragment([dim_k], accum_dtype)
                T.clear(state_dot_sum)
                for i_vb in T.serial(dim_v // state_cols):
                    for i_k, i_v in T.Parallel(dim_k, state_cols):
                        state_dot[i_k, i_v] = (
                            h[i_b, i_c, i_h, i_k, i_vb * state_cols + i_v]
                            * dh[i_b, i_c + 1, i_h, i_k, i_vb * state_cols + i_v]
                        )
                    T.reduce_sum(state_dot, state_dot_sum, dim=1, clear=False)
                for i_k in T.Parallel(dim_k):
                    tail_s[i_k] = state_dot_sum[i_k] * T.exp2(
                        g_cumsum_s[chunk_size - 1, i_k] * LOG2E
                    )

                a = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                factor = T.alloc_fragment([chunk_size, dim_k], accum_dtype)
                T.clear(a)
                if span[0] < BF16_SPLIT_EXP2_SPAN:
                    for i_t, i_k in T.Parallel(chunk_size, dim_k):
                        factor[i_t, i_k] = T.cast(
                            q[i_b, start + i_t, i_h, i_k], accum_dtype
                        ) * T.exp2(g_cumsum_s[i_t, i_k] * LOG2E)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k):
                        q_hi_s[i_t, i_k] = T.cast(factor[i_t, i_k], factor_dtype)
                        if paired:
                            q_lo_s[i_t, i_k] = T.cast(
                                factor[i_t, i_k]
                                - T.cast(T.cast(factor[i_t, i_k], factor_dtype), accum_dtype),
                                factor_dtype,
                            )
                    for i_t, i_k in T.Parallel(chunk_size, dim_k):
                        factor[i_t, i_k] = T.cast(
                            k[i_b, start + i_t, i_h, i_k], accum_dtype
                        ) * T.exp2(-g_cumsum_s[i_t, i_k] * LOG2E)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k):
                        k_hi_s[i_t, i_k] = T.cast(factor[i_t, i_k], factor_dtype)
                        if paired:
                            k_lo_s[i_t, i_k] = T.cast(
                                factor[i_t, i_k]
                                - T.cast(T.cast(factor[i_t, i_k], factor_dtype), accum_dtype),
                                factor_dtype,
                            )
                    T.gemm(q_hi_s, k_hi_s, a, transpose_B=True)
                    if paired:
                        T.gemm(q_hi_s, k_lo_s, a, transpose_B=True)
                        T.gemm(q_lo_s, k_hi_s, a, transpose_B=True)
                else:
                    for i_k in T.Serial(dim_k):
                        for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                            a[i_t, i_j] = a[i_t, i_j] + (
                                T.cast(q[i_b, start + i_t, i_h, i_k], accum_dtype)
                                * T.cast(k[i_b, start + i_j, i_h, i_k], accum_dtype)
                                * T.exp2((g_cumsum_s[i_t, i_k] - g_cumsum_s[i_j, i_k]) * LOG2E)
                            )
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    a_s[i_t, i_j] = T.cast(
                        T.if_then_else(i_j <= i_t, a[i_t, i_j] * scale, 0.0), dtype
                    )

                dv_acc = T.alloc_fragment([chunk_size, dim_v], accum_dtype)
                T.clear(dv_acc)
                T.gemm(a_s, do_s, dv_acc, transpose_A=True)
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    dh_s[i_k, i_v] = T.cast(dh[i_b, i_c + 1, i_h, i_k, i_v], dtype)
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    k_decay_s[i_t, i_k] = T.cast(
                        T.cast(k[i_b, start + i_t, i_h, i_k], accum_dtype)
                        * T.exp2((g_cumsum_s[chunk_size - 1, i_k] - g_cumsum_s[i_t, i_k]) * LOG2E),
                        dtype,
                    )

                T.gemm(k_decay_s, dh_s, dv_acc)
                T.copy(dv_acc, dv[i_b, start : start + chunk_size, i_h, :])

                # The state gradient's share in dk.
                dk_state = T.alloc_fragment([chunk_size, dim_k], accum_dtype)
                T.clear(dk_state)
                T.gemm(v_s, dh_s, dk_state, transpose_B=True)
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    dk_state[i_t, i_k] = dk_state[i_t, i_k] * T.exp2(
                        (g_cumsum_s[chunk_size - 1, i_k] - g_cumsum_s[i_t, i_k]) * LOG2E
                    )

                # dA as a bfloat16 pair.
                da = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                T.clear(da)
                T.gemm(do_s, v_s, da, transpose_B=True)
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    da[i_t, i_j] = T.if_then_else(i_j <= i_t, da[i_t, i_j] * scale, 0.0)
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    da_hi_s[i_t, i_j] = T.cast(da[i_t, i_j], factor_dtype)
                    if paired:
                        da_lo_s[i_t, i_j] = T.cast(
                            da[i_t, i_j] - T.cast(T.cast(da[i_t, i_j], factor_dtype), accum_dtype),
                            factor_dtype,
                        )

                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    h_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v] * scale, dtype)
                dq_acc = T.alloc_fragment([chunk_size, dim_k], accum_dtype)
                T.clear(dq_acc)
                if span[0] < BF16_SPLIT_EXP2_SPAN:
                    T.gemm(da_hi_s, k_hi_s, dq_acc)
                    if paired:
                        T.gemm(da_hi_s, k_lo_s, dq_acc)
                        T.gemm(da_lo_s, k_hi_s, dq_acc)
                T.gemm(do_s, h_s, dq_acc, transpose_B=True)
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    dq_acc[i_t, i_k] = dq_acc[i_t, i_k] * T.exp2(g_cumsum_s[i_t, i_k] * LOG2E)

                dk_acc = T.alloc_fragment([chunk_size, dim_k], accum_dtype)
                T.clear(dk_acc)
                if span[0] < BF16_SPLIT_EXP2_SPAN:
                    T.gemm(da_hi_s, q_hi_s, dk_acc, transpose_A=True)
                    if paired:
                        T.gemm(da_hi_s, q_lo_s, dk_acc, transpose_A=True)
                        T.gemm(da_lo_s, q_hi_s, dk_acc, transpose_A=True)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k):
                        dk_acc[i_t, i_k] = dk_acc[i_t, i_k] * T.exp2(-g_cumsum_s[i_t, i_k] * LOG2E)
                else:
                    for i_j in T.Serial(chunk_size):
                        for i_t, i_k in T.Parallel(chunk_size, dim_k):
                            dq_acc[i_t, i_k] = dq_acc[i_t, i_k] + T.if_then_else(
                                i_j <= i_t,
                                (
                                    T.cast(da_hi_s[i_t, i_j], accum_dtype)
                                    + (T.cast(da_lo_s[i_t, i_j], accum_dtype) if paired else 0.0)
                                )
                                * T.cast(k[i_b, start + i_j, i_h, i_k], accum_dtype)
                                * T.exp2((g_cumsum_s[i_t, i_k] - g_cumsum_s[i_j, i_k]) * LOG2E),
                                0.0,
                            )
                            dk_acc[i_t, i_k] = dk_acc[i_t, i_k] + T.if_then_else(
                                i_t <= i_j,
                                (
                                    T.cast(da_hi_s[i_j, i_t], accum_dtype)
                                    + (T.cast(da_lo_s[i_j, i_t], accum_dtype) if paired else 0.0)
                                )
                                * T.cast(q[i_b, start + i_j, i_h, i_k], accum_dtype)
                                * T.exp2((g_cumsum_s[i_j, i_k] - g_cumsum_s[i_t, i_k]) * LOG2E),
                                0.0,
                            )
                T.copy(dq_acc, dq[i_b, start : start + chunk_size, i_h, :])

                # dg: the upper ones times q dq - k dk_intra (the tail on its last row), plus
                # the strictly lower ones times k dk_state; each operand is formed in place.
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    dq_acc[i_t, i_k] = (
                        T.cast(q[i_b, start + i_t, i_h, i_k], accum_dtype) * dq_acc[i_t, i_k]
                        - T.cast(k[i_b, start + i_t, i_h, i_k], accum_dtype) * dk_acc[i_t, i_k]
                        + T.if_then_else(i_t == chunk_size - 1, tail_s[i_k], 0.0)
                    )
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    dk_acc[i_t, i_k] = dk_acc[i_t, i_k] + dk_state[i_t, i_k]
                T.copy(dk_acc, dk[i_b, start : start + chunk_size, i_h, :])
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    dk_state[i_t, i_k] = (
                        T.cast(k[i_b, start + i_t, i_h, i_k], accum_dtype) * dk_state[i_t, i_k]
                    )

                dg_acc = T.alloc_fragment([chunk_size, dim_k], accum_dtype)
                T.clear(dg_acc)
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    ones_s[i_t, i_j] = T.cast(T.if_then_else(i_t <= i_j, 1.0, 0.0), split_dtype)
                _split_into_ones_product(dq_acc, dg0_s, dg1_s, dg2_s, ones_s, dg_acc, split_dtype)
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    ones_s[i_t, i_j] = T.cast(T.if_then_else(i_j < i_t, 1.0, 0.0), split_dtype)
                _split_into_ones_product(dk_state, dg0_s, dg1_s, dg2_s, ones_s, dg_acc, split_dtype)
                T.copy(dg_acc, dg[i_b, start : start + chunk_size, i_h, :])

        return _main

    return _chunk_func


class GLAChunkBwdKernel(Kernel, GLAChunkBwdInterface):
    """GLA backward kernel: a chunk-parallel state-gradient update, a walk of the state
    gradient with no matmul on its chain, and a chunk-parallel pass for dq, dk, dv and dg.

    h is read from forward's h_out (no recomputation needed).
    """

    supported_archs: list[int] = [80, 89, 90]

    @classmethod
    def refusal(cls, call: GLAChunkCall) -> Optional[str]:
        """Why no program serves this call, or ``None``.

        Refuses a call whose chunk pass needs more shared memory than the device gives.
        """
        reason = head_count_refusal(call.heads) or cls.region_refusal(
            call.dim_k, call.dim_v, call.chunk_size, call.dtype, call.arch
        )
        if reason is not None or not call.smem_budget:
            return reason
        c, k, v = call.chunk_size, call.dim_k, call.dim_v
        need = cls._shared_bytes(c, k, v, call.dtype.itemsize)
        if need <= call.smem_budget:
            return None
        return (
            f"needs at least {need} bytes of shared memory per block at chunk {c}, head dims "
            f"{k} / {v} in {call.dtype}; the device gives {call.smem_budget}"
        )

    @classmethod
    def entry_for(cls, call: GLAChunkCall) -> Entry:
        """``has_initial_state`` is a launch argument, so it is out of the identity."""
        index = call.device.index if call.device is not None else None
        identity = (
            call.batch,
            call.seq_len,
            call.heads,
            call.dim_k,
            call.dim_v,
            call.chunk_size,
            call.scale,
            call.dtype,
            index,
        )
        return identity, lambda: cls(*identity[:-1], device_index=index)

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale if scale > 0 else dim_k**-0.5
        self.dtype = dtype
        self.dtype_name = str(dtype).split(".")[-1]
        reason = self.region_refusal(
            dim_k, dim_v, chunk_size, dtype, get_sm_version(self.device_index)
        )
        if reason:
            raise ValueError(f"{type(self).__name__} does not serve this call: {reason}")
        self.init_config(config)
        self._build_kernels(self.config)

    @staticmethod
    def region_refusal(
        dim_k: int, dim_v: int, chunk_size: int, dtype: torch.dtype, arch: int
    ) -> Optional[str]:
        """Why the default tiling cannot build these extents, or ``None`` when it can.

        The chunk-parallel passes run their GEMMs on four warps, which take whole 16-row
        tiles of ``chunk_size`` rows: 32 splits two by two, and otherwise all four split the
        rows, which takes a multiple of 64. The products put ``dim_k`` and ``dim_v`` on the
        columns. On SM90 a 16-bit ``dim_k`` of at least 64 takes the warp-group instruction,
        which splits four warps over it and admits ``dim_k`` 64 past a multiple of 128; the
        per-warp instruction does not.
        """
        if not (chunk_size == 32 or chunk_size % 64 == 0):
            return f"chunk_size={chunk_size} must be 32 or a multiple of 64"
        warp_group = arch == 90 and torch.finfo(dtype).bits == 16
        served = (
            dim_k == 32
            and dim_v % 128 == 0
            or dim_k == 64
            and dim_v % 64 == 0
            or dim_k % 128 == 0
            and dim_v % 32 == 0
            or warp_group
            and dim_k % 128 == 64
            and dim_v % 64 == 0
        )
        if not served:
            return (
                f"dim_k={dim_k}, dim_v={dim_v}: dim_k must be 32 with dim_v a multiple of "
                "128, 64 with dim_v a multiple of 64, or a multiple of 128 with dim_v a "
                "multiple of 32"
                + (
                    "; or 64 past a multiple of 128 with dim_v a multiple of 64"
                    if warp_group
                    else ""
                )
            )
        return None

    @staticmethod
    def _shared_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Lower bound on the chunk pass's shared memory at chunk *c*, head dims *k* / *v*.

        The fp32 gate cumsum, v, do and the factor tiles (two pairs of bfloat16 tiles, or two
        float32 tiles) stay live from the factors to dk; with them is one of A, the decayed keys
        with the cast state gradient, or the dA pair. Smaller buffers are left out. The update
        pass and the walk need less.
        """
        return c * k * 12 + 2 * c * v * elem + max(c * c * 4, c * c * elem, (c + v) * k * elem)

    @property
    def default_config(self) -> dict:
        """One warp group for the chunk-parallel passes and the walk's prefetch depth."""
        return {"threads": 128, "num_stages": 16}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"threads": threads, "num_stages": stages}
            for threads in (128, 256)
            for stages in (4, 8, 16, 32)
        ]

    def _build_kernels(self, config: dict) -> None:
        """Build the update, walk and chunk passes from a config dict."""
        shape = (self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v, self.chunk_size)
        threads = config["threads"]
        self._update_fn = _gla_bwd_update_kernel(*shape, self.scale, self.dtype_name)(threads)
        rows, stages, walk_threads = state_walk_plan(
            self.batch, self.heads, self.dim_k, self.dim_v, config["num_stages"]
        )
        self._walk_fn = gla_state_walk_kernel(*shape, rows, True, reverse=True)(
            stages, walk_threads
        )
        self._chunk_fn = _gla_bwd_chunk_kernel(*shape, self.scale, self.dtype_name)(threads)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Custom autotuning for multi-kernel backward pass."""
        if self.autotune_configs is None:
            return
        print(
            f"Start autotuning {self.__class__.__name__} ({len(self.autotune_configs)} configs)..."
        )

        B, T, H, K, V = (self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v)
        BT = self.chunk_size
        NT = T // BT
        dtype_torch = self.dtype

        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype_torch).abs()
        h = torch.randn(B, NT + 1, H, K, V, device="cuda", dtype=torch.float32) * 0.01
        do = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        dht = torch.zeros(B, H, K, V, dtype=torch.float32, device="cuda")

        best_lat = float("inf")
        best_cfg = None

        for cfg in self.tuning_candidates(self.autotune_configs):
            try:
                self._build_kernels(cfg)

                # Warmup run
                self.forward(q, k, v, g, h, do, dht)
                torch.cuda.synchronize()

                lat = do_bench(
                    lambda: self.forward(q, k, v, g, h, do, dht),
                    warmup=warmup,
                    rep=rep,
                )
                print(f"  config={cfg} -> {lat:.3f}ms")
                if lat < best_lat:
                    best_lat = lat
                    best_cfg = cfg
            except Exception as e:
                print(f"  config={cfg} -> FAILED: {e}")
                continue

        if best_cfg is not None:
            self.config = best_cfg
            self._build_kernels(best_cfg)
            print(f"Best config: {best_cfg} ({best_lat:.3f}ms)")
        else:
            print("Autotuning failed, using default config")
            self.config = self.default_config
            self._build_kernels(self.config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        has_initial_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, g, do = (t.to(self.dtype) for t in (q, k, v, g, do))
        update, g_last = self._update_fn(q, do, g)
        dh = self._walk_fn(update, g_last, dht.to(torch.float32))
        return self._chunk_fn(q, k, v, g, do, h.to(torch.float32), dh)
