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

from tileops.kernels.online_softmax import (
    LOG2E,
    make_apply_softcap,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

from .call_spec import uses_sliding_window
from .packed_prefill import PackedPrefillKernel
from .prefill_mask import make_bottom_right_attention_mask

__all__ = [
    "GQASlidingWindowVarlenFwdWgmmaPipelinedKernel",
]


class _GQASlidingWindowVarlenFwdKernelBase(PackedPrefillKernel):
    """Shared base for variable-length GQA sliding window forward kernels."""

    @classmethod
    def applies(cls, call) -> bool:
        return uses_sliding_window(call) and not call.is_fp8 and not call.is_uniform

    def _build_program(self) -> None:
        # Specialized on the packed totals, which are known per call.
        self.kernel = None

    @property
    def _accum_dtype_str(self) -> str:
        return "float" if self.accum_dtype == torch.float32 else "double"

    def _call_wrapped(
        self, wrapped_fn, q, k, v, cu_seqlens_q, cu_seqlens_k
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_q, total_k = q.shape[0], k.shape[0]
        return wrapped_fn(
            self.batch,
            self.heads,
            self.heads_kv,
            total_q,
            total_k,
            self.dim,
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self._accum_dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
            self.max_seqlen_q,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
        )


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
    sm_scale: Optional[float] = None,
    softcap: float = 0.0,
    dtype: str = "float16",
    accum_dtype: str = "float",
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    has_window = window_size_left >= 0 or window_size_right >= 0

    @tilelang.jit(
        out_idx=[6, 7],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_sw_fwd_varlen_wgmma_pipelined_func(block_m, block_n, num_stages, threads):
        q_shape = (total_q, heads, dim)
        kv_shape = (total_k, heads_kv, dim)
        apply_mask = make_bottom_right_attention_mask(
            is_causal,
            window_size_left,
            window_size_right,
            accum_dtype,
            block_m,
            block_n,
        )
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n)
            if use_softcap
            else None
        )
        online_softmax = make_online_softmax_with_mask_guard(scale, accum_dtype, block_m, block_n)
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
            T.copy(
                k[kv_start + k_idx * block_n : kv_start + (k_idx + 1) * block_n, by // groups, :],
                k_shared,
            )
            apply_mask(acc_s, k_idx, bx, q_len, kv_len, offset)
            T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

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
            T.copy(
                v[kv_start + k_idx * block_n : kv_start + (k_idx + 1) * block_n, by // groups, :],
                v_shared,
            )
            T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.prim_func
        def _gqa_sw_fwd_varlen_wgmma_pipelined_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_k: T.Tensor([batch + 1], T.int32),  # type: ignore
            max_seqlen_q: T.int32,  # type: ignore
            output: T.Tensor(q_shape, dtype),  # type: ignore
            lse: T.Tensor([heads, total_q], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(max_seqlen_q, block_m), heads, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
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

                T.annotate_layout({o_shared: tilelang.layout.make_swizzled_layout(o_shared)})

                q_start = cu_seqlens_q[bz]
                kv_start = cu_seqlens_k[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                kv_len = cu_seqlens_k[bz + 1] - kv_start
                offset = kv_len - q_len

                T.copy(q[q_start + bx * block_m : q_start + (bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    k_end = T.ceildiv(T.min(kv_len, offset + (bx + 1) * block_m), block_n)
                elif has_window and window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(kv_len, offset + (bx + 1) * block_m + window_size_right), block_n
                    )
                else:
                    k_end = T.ceildiv(kv_len, block_n)

                if has_window and window_size_left >= 0:
                    k_start = T.max(0, offset + bx * block_m - window_size_left) // block_n
                else:
                    k_start = 0

                loop_count = T.max(k_end - k_start, 0)

                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    mma0(
                        k, q_shared, k_shared, acc_s, k_idx, bx, by, kv_start, q_len, kv_len, offset
                    )
                    if use_softcap:
                        apply_softcap(acc_s)
                    online_softmax(
                        acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                    )
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    mma1(v, v_shared, acc_s_cast, acc_o, k_idx, by, kv_start)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] = T.if_then_else(logsum[i] > 0, acc_o[i, j] / logsum[i], 0.0)
                T.sync_threads(3, threads)
                T.copy(acc_o, o_shared)
                T.sync_threads(3, threads)
                for i, j in T.Parallel(block_m, dim):
                    if bx * block_m + i < q_len:
                        output[q_start + bx * block_m + i, by, j] = o_shared[i, j]
                for i in T.Parallel(block_m):
                    if bx * block_m + i < q_len:
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                        lse[by, q_start + bx * block_m + i] = logsum[i]

        return _gqa_sw_fwd_varlen_wgmma_pipelined_main

    return _gqa_sw_fwd_varlen_wgmma_pipelined_func


@torch.library.custom_op("top::gqa_sw_fwd_varlen_wgmma_pipelined_wrapped_kernel", mutates_args=())
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
    sm_scale: float,
    softcap: float,
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
        batch,
        heads,
        heads_kv,
        total_q,
        total_k,
        dim,
        is_causal,
        window_size_left,
        window_size_right,
        sm_scale,
        softcap,
        dtype,
        accum_dtype,
    )(block_m, block_n, num_stages, threads)(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)


@_gqa_sw_fwd_varlen_wgmma_pipelined_wrapped_kernel.register_fake
def _(
    batch,
    heads,
    heads_kv,
    total_q,
    total_k,
    dim,
    is_causal,
    window_size_left,
    window_size_right,
    sm_scale,
    softcap,
    dtype,
    accum_dtype,
    block_m,
    block_n,
    num_stages,
    threads,
    max_seqlen_q,
    q,
    k,
    v,
    cu_q,
    cu_k,
):
    fake_o = torch.empty([total_q, heads, dim], dtype=q.dtype, device=q.device)
    fake_lse = fake_o.new_empty([heads, total_q])
    return fake_o, fake_lse


class GQASlidingWindowVarlenFwdWgmmaPipelinedKernel(_GQASlidingWindowVarlenFwdKernelBase):
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
        configs = list(itertools.product([64, 128], [64, 128], [2, 3], [128, 256]))
        return [
            {"block_m": c[0], "block_n": c[1], "num_stages": c[2], "threads": c[3]} for c in configs
        ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output, _ = self._call_wrapped(
            _gqa_sw_fwd_varlen_wgmma_pipelined_wrapped_kernel, q, k, v, cu_seqlens_q, cu_seqlens_kv
        )
        return output
