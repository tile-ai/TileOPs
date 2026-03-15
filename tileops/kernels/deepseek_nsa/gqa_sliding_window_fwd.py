import itertools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.online_softmax import make_log2e_scale

__all__ = [
    'GqaSlidingWindowFwdKernel',
    'GqaSlidingWindowFwdWgmmaPipelinedKernel',
]


def _gqa_sw_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,   # -1 = unlimited
    window_size_right: int,  # -1 = unlimited
    dtype: str = 'float16',
) -> Callable:
    scale = make_log2e_scale(dim)
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"
    has_window = window_size_left >= 0 or window_size_right >= 0

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_sw_fwd_func(block_m: int, block_n: int, num_stages: int,
                         threads: int) -> Callable:
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        @T.prim_func
        def _gqa_sw_fwd_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            output: T.Tensor(q_shape, dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(seq_len, block_m), heads, batch,
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

                T.copy(q[bz, bx * block_m:(bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # ── Loop range ──────────────────────────────────────────────
                if is_causal:
                    k_end = T.ceildiv(T.min(seq_len, (bx + 1) * block_m), block_n)
                elif has_window and window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(seq_len, (bx + 1) * block_m + window_size_right),
                        block_n)
                else:
                    k_end = T.ceildiv(seq_len, block_n)

                if has_window and window_size_left >= 0:
                    k_start = T.max(0, bx * block_m - window_size_left) // block_n
                else:
                    k_start = 0

                loop_count = T.max(k_end - k_start, 0)

                # ── Main loop ────────────────────────────────────────────────
                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset

                    T.copy(k[bz, k_idx * block_n:(k_idx + 1) * block_n,
                              by // groups, :], k_shared)

                    # Four compile-time mask paths
                    if is_causal and has_window:
                        for i, j in T.Parallel(block_m, block_n):
                            causal_mask = bx * block_m + i < k_idx * block_n + j
                            left_mask = (window_size_left >= 0) and (
                                k_idx * block_n + j < bx * block_m + i - window_size_left)
                            acc_s[i, j] = T.if_then_else(
                                causal_mask or left_mask,
                                -T.infinity(accum_dtype), 0)
                    elif is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_m + i < k_idx * block_n + j,
                                -T.infinity(accum_dtype), 0)
                    elif has_window:
                        for i, j in T.Parallel(block_m, block_n):
                            left_mask = (window_size_left >= 0) and (
                                k_idx * block_n + j < bx * block_m + i - window_size_left)
                            right_mask = (window_size_right >= 0) and (
                                k_idx * block_n + j > bx * block_m + i + window_size_right)
                            acc_s[i, j] = T.if_then_else(
                                left_mask or right_mask,
                                -T.infinity(accum_dtype), 0)
                    else:
                        T.clear(acc_s)

                    T.gemm(q_shared, k_shared, acc_s,
                           transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(v[bz, k_idx * block_n:(k_idx + 1) * block_n,
                              by // groups, :], v_shared)
                    # Online softmax with scores_max clamping.
                    # Clamping prevents exp2(+inf) when all block scores are -inf.
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_m):
                        # Clamp to a finite floor so exp2(prev - curr) never
                        # evaluates to exp2(-inf - (-inf)) = exp2(nan).
                        scores_max[i] = T.max(scores_max[i],
                                              T.cast(-1e38, accum_dtype))
                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                                 scores_max[i] * scale)
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale -
                                             scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, v_shared, acc_o,
                           policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, output[bz, bx * block_m:(bx + 1) * block_m, by, :])
                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_m:(bx + 1) * block_m])

        return _gqa_sw_fwd_main

    return _gqa_sw_fwd_func


@torch.library.custom_op("top::gqa_sw_fwd_wrapped_kernel", mutates_args=())
def _gqa_sw_fwd_wrapped_kernel(
    batch: int, heads: int, heads_kv: int, seq_len: int, dim: int,
    is_causal: bool, window_size_left: int, window_size_right: int,
    dtype: str,
    block_m: int, block_n: int, num_stages: int, threads: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_sw_fwd_kernel(
        batch, heads, heads_kv, seq_len, dim,
        is_causal, window_size_left, window_size_right, dtype)(
        block_m, block_n, num_stages, threads)(q, k, v)


@_gqa_sw_fwd_wrapped_kernel.register_fake
def _(batch, heads, heads_kv, seq_len, dim, is_causal, window_size_left,
      window_size_right, dtype, block_m, block_n, num_stages, threads,
      *inputs):
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class GqaSlidingWindowFwdKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        window_size_left: int,   # -1 = unlimited
        window_size_right: int,  # -1 = unlimited
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype

        self.kernel = _gqa_sw_fwd_kernel(
            self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
            self.is_causal, self.window_size_left, self.window_size_right,
            self.dtype_str)

        self.init_config(config, tune)

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
        configs = list(itertools.product([32, 64, 128], [32, 64, 128],
                                         [1, 2, 3], [128, 256]))
        return [{'block_m': c[0], 'block_n': c[1],
                 'num_stages': c[2], 'threads': c[3]} for c in configs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_sw_fwd_wrapped_kernel(
            self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
            self.is_causal, self.window_size_left, self.window_size_right,
            self.dtype_str,
            self.config["block_m"], self.config["block_n"],
            self.config["num_stages"], self.config["threads"],
            q, k, v)


def _gqa_sw_fwd_wgmma_pipelined_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: str = "float16",
) -> Callable:
    scale = make_log2e_scale(dim)
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"
    has_window = window_size_left >= 0 or window_size_right >= 0

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_sw_fwd_wgmma_pipelined_func(block_m, block_n, num_stages, threads):
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        @T.macro
        def mma0(
            k: T.Tensor(kv_shape, dtype),
            q_shared: T.SharedBuffer([block_m, dim], dtype),
            k_shared: T.SharedBuffer([block_n, dim], dtype),
            acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype),
            k_idx: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ) -> None:
            T.copy(k[bz, k_idx * block_n:(k_idx + 1) * block_n,
                      by // groups, :], k_shared)
            if is_causal and has_window:
                for i, j in T.Parallel(block_m, block_n):
                    causal_mask = bx * block_m + i < k_idx * block_n + j
                    left_mask = (window_size_left >= 0) and (
                        k_idx * block_n + j < bx * block_m + i - window_size_left)
                    acc_s[i, j] = T.if_then_else(
                        causal_mask or left_mask, -T.infinity(accum_dtype), 0)
            elif is_causal:
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_m + i < k_idx * block_n + j,
                        -T.infinity(accum_dtype), 0)
            elif has_window:
                for i, j in T.Parallel(block_m, block_n):
                    left_mask = (window_size_left >= 0) and (
                        k_idx * block_n + j < bx * block_m + i - window_size_left)
                    right_mask = (window_size_right >= 0) and (
                        k_idx * block_n + j > bx * block_m + i + window_size_right)
                    acc_s[i, j] = T.if_then_else(
                        left_mask or right_mask, -T.infinity(accum_dtype), 0)
            else:
                T.clear(acc_s)
            T.gemm(q_shared, k_shared, acc_s,
                   transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def mma1(
            v: T.Tensor(kv_shape, dtype),
            v_shared: T.SharedBuffer([block_n, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype),
            acc_o: T.FragmentBuffer([block_m, dim], accum_dtype),
            k_idx: T.int32,
            by: T.int32,
            bz: T.int32,
        ) -> None:
            T.copy(v[bz, k_idx * block_n:(k_idx + 1) * block_n,
                      by // groups, :], v_shared)
            T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.prim_func
        def _gqa_sw_fwd_wgmma_pipelined_main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            output: T.Tensor(q_shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ) -> None:
            with T.Kernel(
                T.ceildiv(seq_len, block_m), heads, batch,
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
                T.copy(q[bz, bx * block_m:(bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    k_end = T.ceildiv(T.min(seq_len, (bx + 1) * block_m), block_n)
                elif has_window and window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(seq_len, (bx + 1) * block_m + window_size_right),
                        block_n)
                else:
                    k_end = T.ceildiv(seq_len, block_n)

                if has_window and window_size_left >= 0:
                    k_start = T.max(0, bx * block_m - window_size_left) // block_n
                else:
                    k_start = 0

                loop_count = T.max(k_end - k_start, 0)

                for k_offset in T.Pipelined(
                    loop_count,
                    num_stages=num_stages):
                    k_idx = k_start + k_offset
                    mma0(k, q_shared, k_shared, acc_s, k_idx, bx, by, bz)
                    # Online softmax with scores_max clamping.
                    # Clamping prevents exp2(+inf) when all block scores are -inf.
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_m):
                        # Clamp to a finite floor so exp2(prev - curr) never
                        # evaluates to exp2(-inf - (-inf)) = exp2(nan).
                        scores_max[i] = T.max(scores_max[i],
                                              T.cast(-1e38, accum_dtype))
                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                                 scores_max[i] * scale)
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale -
                                             scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]
                    mma1(v, v_shared, acc_s_cast, acc_o, k_idx, by, bz)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, output[bz, bx * block_m:(bx + 1) * block_m, by, :])
                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_m:(bx + 1) * block_m])

        return _gqa_sw_fwd_wgmma_pipelined_main

    return _gqa_sw_fwd_wgmma_pipelined_func


@torch.library.custom_op(
    "top::gqa_sw_fwd_wgmma_pipelined_wrapped_kernel", mutates_args=())
def _gqa_sw_fwd_wgmma_pipelined_wrapped_kernel(
    batch: int, heads: int, heads_kv: int, seq_len: int, dim: int,
    is_causal: bool, window_size_left: int, window_size_right: int,
    dtype: str,
    block_m: int, block_n: int, num_stages: int, threads: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_sw_fwd_wgmma_pipelined_kernel(
        batch, heads, heads_kv, seq_len, dim,
        is_causal, window_size_left, window_size_right, dtype)(
        block_m, block_n, num_stages, threads)(q, k, v)


@_gqa_sw_fwd_wgmma_pipelined_wrapped_kernel.register_fake
def _(batch, heads, heads_kv, seq_len, dim, is_causal, window_size_left,
      window_size_right, dtype, block_m, block_n, num_stages, threads,
      *inputs):
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class GqaSlidingWindowFwdWgmmaPipelinedKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype

        self.kernel = _gqa_sw_fwd_wgmma_pipelined_kernel(
            self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
            self.is_causal, self.window_size_left, self.window_size_right,
            self.dtype_str)

        self.init_config(config, tune)

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
        configs = list(itertools.product([64, 128], [64, 128],
                                         [2, 3], [128, 256]))
        return [{'block_m': c[0], 'block_n': c[1],
                 'num_stages': c[2], 'threads': c[3]} for c in configs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_sw_fwd_wgmma_pipelined_wrapped_kernel(
            self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
            self.is_causal, self.window_size_left, self.window_size_right,
            self.dtype_str,
            self.config["block_m"], self.config["block_n"],
            self.config["num_stages"], self.config["threads"],
            q, k, v)
