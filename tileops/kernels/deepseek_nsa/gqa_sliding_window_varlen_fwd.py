"""Variable-length GQA forward with sliding window attention.

Supports seqlen_q != seqlen_k per sample (e.g. KV-cache prefill).
Inputs are packed (no padding); cu_seqlens arrays record per-sample boundaries.

Mask semantics (FA3-compatible, bottom-right causal alignment):
  offset = seqlen_k - seqlen_q  (per sample)
  causal:      k_pos > q_pos + offset  → masked
  window left: k_pos < q_pos + offset - window_size_left  → masked
  window right:k_pos > q_pos + offset + window_size_right → masked
"""
import functools
import itertools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.online_softmax import (
    make_log2e_scale,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

__all__ = [
    'GqaSlidingWindowVarlenFwdKernel',
    'GqaSlidingWindowVarlenFwdWgmmaPipelinedKernel',
]


def _make_apply_mask(is_causal, has_window, window_size_left, window_size_right,
                     accum_dtype, block_m, block_n):
    """Create a masked attention score initialization macro.

    All parameters are compile-time constants baked into the macro via closure.
    The macro writes 0 or ``-infinity`` into ``acc_s`` depending on the mask
    conditions, using four compile-time paths:

    - causal + window (left only)
    - causal only
    - window only (left and/or right)
    - no masking (OOB guard only)

    Args:
        is_causal: Whether causal masking is applied.
        has_window: Whether any window constraint is active.
        window_size_left: Left window size (-1 = unlimited).
        window_size_right: Right window size (-1 = unlimited).
        accum_dtype: Accumulator data type string (e.g. "float").
        block_m: Tile size along the Q dimension.
        block_n: Tile size along the KV dimension.

    Returns:
        apply_mask: A ``T.macro`` that fills ``acc_s`` with mask values.
    """

    @T.macro
    def apply_mask(acc_s, k_idx, bx, q_len, kv_len, offset):
        if is_causal and has_window:
            for i, j in T.Parallel(block_m, block_n):
                causal_mask = (
                    k_idx * block_n + j > bx * block_m + i + offset)
                left_mask = (window_size_left >= 0) and (
                    k_idx * block_n + j <
                    bx * block_m + i + offset - window_size_left)
                q_oob = bx * block_m + i >= q_len
                k_oob = k_idx * block_n + j >= kv_len
                acc_s[i, j] = T.if_then_else(
                    causal_mask or left_mask or q_oob or k_oob,
                    -T.infinity(accum_dtype), 0)
        elif is_causal:
            for i, j in T.Parallel(block_m, block_n):
                causal_mask = (
                    k_idx * block_n + j > bx * block_m + i + offset)
                q_oob = bx * block_m + i >= q_len
                k_oob = k_idx * block_n + j >= kv_len
                acc_s[i, j] = T.if_then_else(
                    causal_mask or q_oob or k_oob,
                    -T.infinity(accum_dtype), 0)
        elif has_window:
            for i, j in T.Parallel(block_m, block_n):
                left_mask = (window_size_left >= 0) and (
                    k_idx * block_n + j <
                    bx * block_m + i + offset - window_size_left)
                right_mask = (window_size_right >= 0) and (
                    k_idx * block_n + j >
                    bx * block_m + i + offset + window_size_right)
                q_oob = bx * block_m + i >= q_len
                k_oob = k_idx * block_n + j >= kv_len
                acc_s[i, j] = T.if_then_else(
                    left_mask or right_mask or q_oob or k_oob,
                    -T.infinity(accum_dtype), 0)
        else:
            for i, j in T.Parallel(block_m, block_n):
                q_oob = bx * block_m + i >= q_len
                k_oob = k_idx * block_n + j >= kv_len
                acc_s[i, j] = T.if_then_else(
                    q_oob or k_oob, -T.infinity(accum_dtype), 0)

    return apply_mask


@functools.lru_cache(maxsize=None)
def _gqa_sw_fwd_varlen_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,       # total Q tokens packed across all batch samples
    total_k: int,       # total K/V tokens packed across all batch samples
    dim: int,
    is_causal: bool,
    window_size_left: int,   # -1 = unlimited
    window_size_right: int,  # -1 = unlimited
    dtype: str = 'float16',
    accum_dtype: str = 'float',
) -> Callable:
    scale = make_log2e_scale(dim)
    assert heads % heads_kv == 0
    groups = heads // heads_kv
    has_window = window_size_left >= 0 or window_size_right >= 0

    @tilelang.jit(
        out_idx=[6, 7],  # output, lse
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_sw_fwd_varlen_func(block_m: int, block_n: int, num_stages: int,
                                threads: int) -> Callable:
        q_shape = (total_q, heads, dim)
        kv_shape = (total_k, heads_kv, dim)
        apply_mask = _make_apply_mask(
            is_causal, has_window, window_size_left, window_size_right,
            accum_dtype, block_m, block_n)
        online_softmax = make_online_softmax_with_mask_guard(
            scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _gqa_sw_fwd_varlen_main(
            q: T.Tensor(q_shape, dtype),           # type: ignore
            k: T.Tensor(kv_shape, dtype),           # type: ignore
            v: T.Tensor(kv_shape, dtype),           # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_k: T.Tensor([batch + 1], T.int32),  # type: ignore
            max_seqlen_q: T.int32,                  # type: ignore  (grid sizing)
            output: T.Tensor(q_shape, dtype),       # type: ignore
            lse: T.Tensor([heads, total_q], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(max_seqlen_q, block_m), heads, batch,
                threads=threads) as (bx, by, bz):

                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                # ── Per-sample boundaries ─────────────────────────────────────
                # cu_seqlens[bz] .. cu_seqlens[bz+1] is sample bz's token range.
                # offset = kv_len - q_len aligns the causal mask to the
                # bottom-right corner of the attention matrix (FA3 convention).
                #   offset=0  → standard prefill (seqlen_q == seqlen_k)
                #   offset>0  → KV-cache scenario (seqlen_k > seqlen_q)
                q_start = cu_seqlens_q[bz]
                kv_start = cu_seqlens_k[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                kv_len = cu_seqlens_k[bz + 1] - kv_start
                offset = kv_len - q_len

                T.copy(q[q_start + bx * block_m:q_start + (bx + 1) * block_m,
                          by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # ── Loop range (runtime seqlens + offset) ─────────────────────
                if is_causal:
                    k_end = T.ceildiv(
                        T.min(kv_len, offset + (bx + 1) * block_m), block_n)
                elif has_window and window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(kv_len,
                              offset + (bx + 1) * block_m + window_size_right),
                        block_n)
                else:
                    k_end = T.ceildiv(kv_len, block_n)

                if has_window and window_size_left >= 0:
                    k_start = T.max(
                        0, offset + bx * block_m - window_size_left) // block_n
                else:
                    k_start = 0

                loop_count = T.max(k_end - k_start, 0)

                # ── Main loop ─────────────────────────────────────────────────
                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset

                    T.copy(k[kv_start + k_idx * block_n:
                               kv_start + (k_idx + 1) * block_n,
                              by // groups, :], k_shared)
                    apply_mask(acc_s, k_idx, bx, q_len, kv_len, offset)
                    T.gemm(q_shared, k_shared, acc_s,
                           transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(v[kv_start + k_idx * block_n:
                               kv_start + (k_idx + 1) * block_n,
                              by // groups, :], v_shared)
                    online_softmax(acc_s, scores_max, scores_max_prev,
                                   scores_scale, scores_sum, logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, v_shared, acc_o,
                           policy=T.GemmWarpPolicy.FullRow)

                # ── Write back with Q boundary guard ─────────────────────────
                # Guard is required because the grid may launch blocks for
                # bx * block_m >= q_len (short samples in a padded grid).
                for i, j in T.Parallel(block_m, dim):
                    if bx * block_m + i < q_len:
                        output[q_start + bx * block_m + i, by,
                               j] = acc_o[i, j] / logsum[i]
                for i in T.Parallel(block_m):
                    if bx * block_m + i < q_len:
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                        lse[by, q_start + bx * block_m + i] = logsum[i]

        return _gqa_sw_fwd_varlen_main

    return _gqa_sw_fwd_varlen_func


@torch.library.custom_op("top::gqa_sw_fwd_varlen_wrapped_kernel",
                          mutates_args=())
def _gqa_sw_fwd_varlen_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_k: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: str,
    accum_dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    max_seqlen_q: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_sw_fwd_varlen_kernel(
        batch, heads, heads_kv, total_q, total_k, dim,
        is_causal, window_size_left, window_size_right, dtype, accum_dtype)(
        block_m, block_n, num_stages, threads)(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)


@_gqa_sw_fwd_varlen_wrapped_kernel.register_fake
def _(batch, heads, heads_kv, total_q, total_k, dim, is_causal,
      window_size_left, window_size_right, dtype, accum_dtype, block_m, block_n,
      num_stages, threads, max_seqlen_q, q, k, v, cu_q, cu_k):
    fake_o = torch.empty([total_q, heads, dim], dtype=q.dtype, device=q.device)
    fake_lse = fake_o.new_empty([heads, total_q])
    return fake_o, fake_lse


class _GqaSlidingWindowVarlenFwdKernelBase(Kernel):
    """Shared base for variable-length GQA sliding window forward kernels."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        # total_q / total_k are not known at init; resolved per forward() call.
        self.init_config(config, tune)

    @property
    def _accum_dtype_str(self) -> str:
        return "float" if self.accum_dtype == torch.float32 else "double"

    def _call_wrapped(self, wrapped_fn, q, k, v, cu_seqlens_q, cu_seqlens_k,
                      max_seqlen_q) -> Tuple[torch.Tensor, torch.Tensor]:
        total_q, total_k = q.shape[0], k.shape[0]
        return wrapped_fn(
            self.batch, self.heads, self.heads_kv, total_q, total_k, self.dim,
            self.is_causal, self.window_size_left, self.window_size_right,
            self.dtype_str, self._accum_dtype_str,
            self.config["block_m"], self.config["block_n"],
            self.config["num_stages"], self.config["threads"],
            max_seqlen_q,
            q, k, v, cu_seqlens_q, cu_seqlens_k)


class GqaSlidingWindowVarlenFwdKernel(_GqaSlidingWindowVarlenFwdKernelBase):
    """Variable-length GQA sliding window forward kernel (sm80/89/90)."""
    supported_archs: list[int] = [80, 89, 90]

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 64 if self.dim <= 128 else 32,
            "num_stages": 1,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = list(
            itertools.product([32, 64, 128], [32, 64, 128], [1, 2, 3],
                              [128, 256]))
        return [{'block_m': c[0], 'block_n': c[1], 'num_stages': c[2],
                 'threads': c[3]} for c in configs]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._call_wrapped(
            _gqa_sw_fwd_varlen_wrapped_kernel,
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)


# ── WGMMA Pipelined (SM90) ────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _gqa_sw_fwd_varlen_wgmma_pipelined_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_k: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
) -> Callable:
    scale = make_log2e_scale(dim)
    assert heads % heads_kv == 0
    groups = heads // heads_kv
    has_window = window_size_left >= 0 or window_size_right >= 0

    @tilelang.jit(
        out_idx=[6, 7],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_sw_fwd_varlen_wgmma_pipelined_func(block_m, block_n, num_stages,
                                                 threads):
        q_shape = (total_q, heads, dim)
        kv_shape = (total_k, heads_kv, dim)
        apply_mask = _make_apply_mask(
            is_causal, has_window, window_size_left, window_size_right,
            accum_dtype, block_m, block_n)
        online_softmax = make_online_softmax_with_mask_guard(
            scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.macro
        def mma0(
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            q_shared: T.SharedBuffer([block_m, dim], dtype),  # type: ignore
            k_shared: T.SharedBuffer([block_n, dim], dtype),  # type: ignore
            acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype),  # type: ignore
            k_idx: T.int32,  # type: ignore
            bx: T.int32,  # type: ignore
            by: T.int32,  # type: ignore
            kv_start: T.int32,  # type: ignore
            q_len: T.int32,  # type: ignore
            kv_len: T.int32,  # type: ignore
            offset: T.int32,  # type: ignore
        ) -> None:
            T.copy(k[kv_start + k_idx * block_n:
                       kv_start + (k_idx + 1) * block_n,
                      by // groups, :], k_shared)
            apply_mask(acc_s, k_idx, bx, q_len, kv_len, offset)
            T.gemm(q_shared, k_shared, acc_s,
                   transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def mma1(
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            v_shared: T.SharedBuffer([block_n, dim], dtype),  # type: ignore
            acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype),  # type: ignore
            acc_o: T.FragmentBuffer([block_m, dim], accum_dtype),  # type: ignore
            k_idx: T.int32,  # type: ignore
            by: T.int32,  # type: ignore
            kv_start: T.int32,  # type: ignore
        ) -> None:
            T.copy(v[kv_start + k_idx * block_n:
                       kv_start + (k_idx + 1) * block_n,
                      by // groups, :], v_shared)
            T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.prim_func
        def _gqa_sw_fwd_varlen_wgmma_pipelined_main(
            q: T.Tensor(q_shape, dtype),           # type: ignore
            k: T.Tensor(kv_shape, dtype),           # type: ignore
            v: T.Tensor(kv_shape, dtype),           # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_k: T.Tensor([batch + 1], T.int32),  # type: ignore
            max_seqlen_q: T.int32,                  # type: ignore
            output: T.Tensor(q_shape, dtype),       # type: ignore
            lse: T.Tensor([heads, total_q], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(max_seqlen_q, block_m), heads, batch,
                threads=threads) as (bx, by, bz):

                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                o_shared = T.alloc_shared([block_m, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.annotate_layout(
                    {o_shared: tilelang.layout.make_swizzled_layout(o_shared)})

                q_start = cu_seqlens_q[bz]
                kv_start = cu_seqlens_k[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                kv_len = cu_seqlens_k[bz + 1] - kv_start
                offset = kv_len - q_len

                T.copy(q[q_start + bx * block_m:q_start + (bx + 1) * block_m,
                          by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    k_end = T.ceildiv(
                        T.min(kv_len, offset + (bx + 1) * block_m), block_n)
                elif has_window and window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(kv_len,
                              offset + (bx + 1) * block_m + window_size_right),
                        block_n)
                else:
                    k_end = T.ceildiv(kv_len, block_n)

                if has_window and window_size_left >= 0:
                    k_start = T.max(
                        0, offset + bx * block_m - window_size_left) // block_n
                else:
                    k_start = 0

                loop_count = T.max(k_end - k_start, 0)

                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    mma0(k, q_shared, k_shared, acc_s, k_idx, bx, by,
                         kv_start, q_len, kv_len, offset)
                    online_softmax(acc_s, scores_max, scores_max_prev,
                                   scores_scale, scores_sum, logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    mma1(v, v_shared, acc_s_cast, acc_o, k_idx, by, kv_start)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, o_shared)
                for i, j in T.Parallel(block_m, dim):
                    if bx * block_m + i < q_len:
                        output[q_start + bx * block_m + i, by,
                               j] = o_shared[i, j]
                for i in T.Parallel(block_m):
                    if bx * block_m + i < q_len:
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                        lse[by, q_start + bx * block_m + i] = logsum[i]

        return _gqa_sw_fwd_varlen_wgmma_pipelined_main

    return _gqa_sw_fwd_varlen_wgmma_pipelined_func


@torch.library.custom_op(
    "top::gqa_sw_fwd_varlen_wgmma_pipelined_wrapped_kernel", mutates_args=())
def _gqa_sw_fwd_varlen_wgmma_pipelined_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_k: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: str,
    accum_dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    max_seqlen_q: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_sw_fwd_varlen_wgmma_pipelined_kernel(
        batch, heads, heads_kv, total_q, total_k, dim,
        is_causal, window_size_left, window_size_right, dtype, accum_dtype)(
        block_m, block_n, num_stages, threads)(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)


@_gqa_sw_fwd_varlen_wgmma_pipelined_wrapped_kernel.register_fake
def _(batch, heads, heads_kv, total_q, total_k, dim, is_causal,
      window_size_left, window_size_right, dtype, accum_dtype, block_m, block_n,
      num_stages, threads, max_seqlen_q, q, k, v, cu_q, cu_k):
    fake_o = torch.empty([total_q, heads, dim], dtype=q.dtype, device=q.device)
    fake_lse = fake_o.new_empty([heads, total_q])
    return fake_o, fake_lse


class GqaSlidingWindowVarlenFwdWgmmaPipelinedKernel(_GqaSlidingWindowVarlenFwdKernelBase):
    """Variable-length GQA sliding window forward kernel, WGMMA pipelined (sm90)."""
    supported_archs: list[int] = [90]

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "num_stages": 3,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = list(
            itertools.product([64, 128], [64, 128], [2, 3], [128, 256]))
        return [{'block_m': c[0], 'block_n': c[1], 'num_stages': c[2],
                 'threads': c[3]} for c in configs]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._call_wrapped(
            _gqa_sw_fwd_varlen_wgmma_pipelined_wrapped_kernel,
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
