import functools
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

from ._config import tile_stage_thread_configs
from .call_spec import dense_prefill_region
from .packed_prefill import PackedPrefillKernel
from .prefill_mask import make_bottom_right_attention_mask

__all__ = ["GQAPrefillFwdKernel"]

_FAST_COMPILE_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DENABLE_BF16",
]


def _make_apply_softcap_no_mask_guard(score_scale, softcap, accum_dtype, block_rows, block_cols):
    @T.macro
    def apply_softcap(acc_s):
        for i, j in T.Parallel(block_rows, block_cols):
            acc_s[i, j] = T.cast(softcap, accum_dtype) * T.tanh(
                acc_s[i, j] * T.cast(score_scale / softcap, accum_dtype)
            )

    return apply_softcap


# GQA


@functools.lru_cache(maxsize=32)
def _gqa_prefill_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: float = 0.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    dtype: str = "float16",
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if is_causal and seq_len_q > seq_len_kv:
        raise ValueError("causal prefill requires seq_len_q <= seq_len_kv")
    groups = heads // heads_kv
    causal_offset = seq_len_kv - seq_len_q
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_prefill_fwd_func(
        block_m: int, block_n: int, num_stages: int, threads: int
    ) -> Callable:
        q_shape = (batch, seq_len_q, heads, dim)
        kv_shape = (batch, seq_len_kv, heads_kv, dim)
        o_shape = (batch, seq_len_q, heads, dim)
        online_softmax = make_online_softmax_with_mask_guard(scale, accum_dtype, block_m, block_n)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n)
            if use_softcap
            else None
        )
        apply_mask = make_bottom_right_attention_mask(
            is_causal,
            window_size_left,
            window_size_right,
            accum_dtype,
            block_m,
            block_n,
        )
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _gqa_prefill_fwd_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            output: T.Tensor(o_shape, dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len_q], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(seq_len_q, block_m), heads, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
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

                if (bx + 1) * block_m <= seq_len_q:
                    T.copy(
                        q[bz, bx * block_m : (bx + 1) * block_m, by, :], q_shared, disable_tma=True
                    )
                else:
                    for i, d in T.Parallel(block_m, dim):
                        q_pos = bx * block_m + i
                        if q_pos < seq_len_q:
                            q_shared[i, d] = q[bz, q_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, dtype)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    k_end = T.ceildiv(
                        T.min(seq_len_kv, (bx + 1) * block_m + causal_offset),
                        block_n,
                    )
                elif window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(
                            seq_len_kv,
                            (bx + 1) * block_m + causal_offset + window_size_right,
                        ),
                        block_n,
                    )
                else:
                    k_end = T.ceildiv(seq_len_kv, block_n)

                if window_size_left >= 0:
                    k_start = T.max(0, bx * block_m + causal_offset - window_size_left) // block_n
                else:
                    k_start = 0
                loop_range = T.max(k_end - k_start, 0)

                for k_offset in T.Pipelined(loop_range, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    if (k_idx + 1) * block_n <= seq_len_kv:
                        T.copy(
                            k[bz, k_idx * block_n : (k_idx + 1) * block_n, by // groups, :],
                            k_shared,
                            disable_tma=True,
                        )
                        T.copy(
                            v[bz, k_idx * block_n : (k_idx + 1) * block_n, by // groups, :],
                            v_shared,
                            disable_tma=True,
                        )
                    else:
                        for j, d in T.Parallel(block_n, dim):
                            kv_pos = k_idx * block_n + j
                            if kv_pos < seq_len_kv:
                                k_shared[j, d] = k[bz, kv_pos, by // groups, d]
                                v_shared[j, d] = v[bz, kv_pos, by // groups, d]
                            else:
                                k_shared[j, d] = T.cast(0, dtype)
                                v_shared[j, d] = T.cast(0, dtype)
                    apply_mask(
                        acc_s,
                        k_idx,
                        bx,
                        seq_len_q,
                        seq_len_kv,
                        causal_offset,
                    )
                    T.gemm(
                        q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    if use_softcap:
                        apply_softcap(acc_s)
                    online_softmax(
                        acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                    )
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if (bx + 1) * block_m <= seq_len_q:
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] /= logsum[i]
                    T.copy(
                        acc_o,
                        output[bz, bx * block_m : (bx + 1) * block_m, by, :],
                        disable_tma=True,
                    )
                    for i in T.Parallel(block_m):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                    T.copy(logsum, lse[bz, by, bx * block_m : (bx + 1) * block_m], disable_tma=True)
                else:
                    for i, j in T.Parallel(block_m, dim):
                        q_pos = bx * block_m + i
                        if q_pos < seq_len_q:
                            output[bz, q_pos, by, j] = acc_o[i, j] / logsum[i]
                    for i in T.Parallel(block_m):
                        q_pos = bx * block_m + i
                        if q_pos < seq_len_q:
                            lse[bz, by, q_pos] = T.log2(logsum[i]) + scores_max[i] * scale

        return _gqa_prefill_fwd_main

    return _gqa_prefill_fwd_func


@torch.library.custom_op("top::gqa_prefill_fwd_wrapped_kernel", mutates_args=())
def _gqa_prefill_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_prefill_fwd_kernel(
        batch,
        heads,
        heads_kv,
        seq_len_q,
        seq_len_kv,
        dim,
        is_causal,
        sm_scale,
        softcap,
        window_size_left,
        window_size_right,
        dtype,
    )(block_m, block_n, num_stages, threads)(q, k, v)


@_gqa_prefill_fwd_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    *inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len_q])
    return fake_o, fake_lse


class GQAPrefillFwdKernel(PackedPrefillKernel):
    """General dense packed prefill: any head dim, causal or not, fp16/bf16.

    Serves the whole dense region, and is marked ``general`` so that where a
    specialised implementation of this key also applies, that one runs instead.
    """

    supported_archs: list[int] = [80, 89, 90]
    #: The implementation behind the specialised ones for this key.
    general: bool = True

    @classmethod
    def applies(cls, call) -> bool:
        return dense_prefill_region(call)

    def _build_program(self) -> None:
        self.kernel = _gqa_prefill_fwd_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.max_seqlen_q,
            self.max_seqlen_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.window_size_left,
            self.window_size_right,
            self.dtype_str,
        )

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
        return tile_stage_thread_configs()

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
        q_bshd, k_bshd, v_bshd = self._bshd(q, k, v)
        output, _ = _gqa_prefill_fwd_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.max_seqlen_q,
            self.max_seqlen_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.window_size_left,
            self.window_size_right,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
            q_bshd,
            k_bshd,
            v_bshd,
        )
        return output.reshape(q.shape)
