import torch
from typing import Optional, Any, Callable
import itertools
import tilelang
from tilelang import language as T

from top.kernels.kernel import Kernel


def _gqa_window_sliding_kernel(
    batch_size: int,
    groups: int,
    uq: int,
    ukv: int,
    heads: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    scale = (1.0 / dim)**0.5 * 1.44269504
    head_kv = heads // groups
    has_window = window_size_left >= 0 or window_size_right >= 0

    @tilelang.jit(
        out_idx=[6],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _gqa_window_sliding_func(block_m: int, block_n: int, num_stages: int, threads: int):
        q_shape = [uq, heads, dim]
        kv_shape = [ukv, head_kv, dim]
        o_shape = [uq, heads, dim]

        @T.prim_func
        def _parallel_gqa_window_sliding_main(
                q_unpad: T.Tensor(q_shape, dtype),
                k_unpad: T.Tensor(kv_shape, dtype),
                v_unpad: T.Tensor(kv_shape, dtype),
                cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
                cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
                max_seqlen_q: T.int32,
                output_unpad: T.Tensor(o_shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(max_seqlen_q, block_m), heads, batch_size,
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

                batch_idx = bz
                head_idx = by
                kv_head_idx = head_idx // groups

                q_start_idx = cu_seqlens_q[batch_idx]
                kv_start_idx = cu_seqlens_k[batch_idx]
                q_end_idx = cu_seqlens_q[batch_idx + 1]
                k_end_idx = cu_seqlens_k[batch_idx + 1]

                q_current_seqlen = q_end_idx - q_start_idx
                kv_current_seqlen = k_end_idx - kv_start_idx

                T.copy(
                    q_unpad[q_start_idx + bx * block_m:q_start_idx + (bx + 1) * block_m,
                            head_idx, :], q_shared)

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                offset = kv_current_seqlen - q_current_seqlen

                if is_causal:
                    max_visible_k_idx = offset + (bx + 1) * block_m
                    if has_window and window_size_left >= 0:
                        loop_range = T.min(
                            T.ceildiv(max_visible_k_idx, block_n),
                            T.ceildiv(kv_current_seqlen, block_n))
                    else:
                        loop_range = T.min(
                            T.ceildiv(max_visible_k_idx, block_n),
                            T.ceildiv(kv_current_seqlen, block_n))
                else:
                    loop_range = T.ceildiv(kv_current_seqlen, block_n)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        k_unpad[kv_start_idx + k * block_n:kv_start_idx + (k + 1) * block_n,
                                kv_head_idx, :], k_shared)

                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            causal_mask = (bx * block_m + i + offset < k * block_n + j)

                            window_mask_left = T.if_then_else(
                                has_window and window_size_left >= 0, (k * block_n + j)
                                < (bx * block_m + i + offset - window_size_left), (k * block_n + j)
                                < -1)

                            boundary_mask = (
                                bx * block_m + i >= q_current_seqlen or
                                k * block_n + j >= kv_current_seqlen)

                            acc_s[i, j] = T.if_then_else(
                                causal_mask or window_mask_left or boundary_mask,
                                -1e9,
                                0,
                            )
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            window_mask_left = T.if_then_else(
                                has_window and window_size_left >= 0, (k * block_n + j)
                                < (bx * block_m + i + offset - window_size_left), (k * block_n + j)
                                < -1)
                            window_mask_right = T.if_then_else(
                                has_window and window_size_right >= 0, (k * block_n + j)
                                > (bx * block_m + i + offset + window_size_right), (k * block_n + j)
                                < -1)

                            boundary_mask = (
                                bx * block_m + i >= q_current_seqlen or
                                k * block_n + j >= kv_current_seqlen)

                            acc_s[i, j] = T.if_then_else(
                                window_mask_left or window_mask_right or boundary_mask,
                                -1e9,
                                0,
                            )

                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(
                        v_unpad[kv_start_idx + k * block_n:kv_start_idx + (k + 1) * block_n,
                                kv_head_idx, :], v_shared)

                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] = 0 if is_causal and bx * block_m + i + offset < 0 else acc_o[
                        i, j] / logsum[i]

                T.copy(acc_o, o_shared)
                for i, d in T.Parallel(block_m, dim):
                    if bx * block_m + i < q_current_seqlen:
                        output_unpad[q_start_idx + bx * block_m + i, head_idx, d] = o_shared[i, d]

        return _parallel_gqa_window_sliding_main

    return _gqa_window_sliding_func


@torch.library.custom_op("top::gqa_window_sliding_wrapped_kernel", mutates_args=())
def _gqa_window_sliding_wrapped_kernel(
    batch_size: int,
    groups: int,
    uq: int,
    ukv: int,
    heads: int,
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
    q_unpad: torch.Tensor,
    k_unpad: torch.Tensor,
    v_unpad: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    return _gqa_window_sliding_kernel(batch_size, groups, uq, ukv, heads, dim, is_causal,
                                      window_size_left, window_size_right, dtype,
                                      accum_dtype)(block_m, block_n, num_stages,
                                                   threads)(q_unpad, k_unpad, v_unpad, cu_seqlens_q,
                                                            cu_seqlens_k, max_seqlen_q)


@_gqa_window_sliding_wrapped_kernel.register_fake
def _(
    batch_size: int,
    groups: int,
    uq: int,
    ukv: int,
    heads: int,
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
    *inputs: tuple[Any],
) -> torch.Tensor:
    _ = (batch_size, groups, uq, ukv, heads, dim, is_causal, window_size_left, window_size_right,
         dtype, accum_dtype, block_m, block_n, num_stages, threads)
    return torch.empty([uq, heads, dim], dtype=inputs[0].dtype, device=inputs[0].device)


class GQAWindowSlidingKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch_size: int,
                 groups: int,
                 uq: int,
                 ukv: int,
                 heads: int,
                 dim: int,
                 is_causal: bool,
                 window_size_left: int,
                 window_size_right: int,
                 dtype: torch.dtype,
                 accum_dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.groups = groups
        self.uq = uq
        self.ukv = ukv
        self.heads = heads
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.dtype_name = str(dtype).split('.')[-1]
        self.accum_dtype_name = str(accum_dtype).split('.')[-1]

        self.kernel = _gqa_window_sliding_kernel(self.batch_size, self.groups, self.uq, self.ukv,
                                                 self.heads, self.dim, self.is_causal,
                                                 self.window_size_left, self.window_size_right,
                                                 self.dtype_name, self.accum_dtype_name)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "num_stages": 2,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [64, 128]
        block_n = [64, 128]
        num_stages = [1]
        threads = [128]
        _configs = list(itertools.product(block_m, block_n, num_stages, threads))
        return [{
            "block_m": c[0],
            "block_n": c[1],
            "num_stages": c[2],
            "threads": c[3]
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor,
                cu_seqlens_k: torch.Tensor, max_seqlen_q: int) -> torch.Tensor:
        return _gqa_window_sliding_wrapped_kernel(
            self.batch_size, self.groups, self.uq, self.ukv, self.heads, self.dim, self.is_causal,
            self.window_size_left, self.window_size_right, self.dtype_name, self.accum_dtype_name,
            self.config["block_m"], self.config["block_n"], self.config["num_stages"],
            self.config["threads"], q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
