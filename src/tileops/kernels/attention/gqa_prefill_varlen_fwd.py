"""Packed variable-length GQA prefill forward kernel.

Inputs use THD layout:
  q: [T_q, H, D]
  k/v: [T_kv, H_kv, D]

``cu_seqlens_q`` and ``cu_seqlens_kv`` describe per-request packed ranges.
Causal masking uses bottom-right alignment per request, matching the dense
prefill contract when q_len may be smaller than kv_len.
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

from .fp8_prefill_core import make_native_fp8_prefill_tile_update
from .packed_prefill import PackedPrefillKernel
from .prefill_mask import make_bottom_right_attention_mask
from .prefill_rope import make_prefill_rope_policy

__all__ = [
    "GQAPrefillVarlenFP8TensorCoreFwdKernel",
    "GQAPrefillVarlenFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _gqa_prefill_varlen_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: float = 0.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    input_dtype: str = "float16",
    output_dtype: str = "float16",
    fuse_rope: bool = False,
    max_position: int = 1,
    rotary_dim: int = 0,
    rope_layout: str = "neox",
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    native_fp8 = input_dtype == "float8_e4m3fn"
    softmax_scale = LOG2E if native_fp8 or use_softcap else score_scale * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if fuse_rope and (
        rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > dim
    ):
        raise ValueError("rotary_dim must be positive, even, and <= dim")
    if rope_layout not in ("neox", "interleaved"):
        raise ValueError("rope_layout must be 'neox' or 'interleaved'")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[12, 13],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_prefill_varlen_fwd_func(
        block_m: int, block_n: int, num_stages: int, threads: int
    ) -> Callable:
        q_shape = (total_q, heads, dim)
        kv_shape = (total_kv, heads_kv, dim)
        scale_shape = (batch, heads_kv)
        rope_cols, paired_dim, rotate = make_prefill_rope_policy(
            fuse_rope, rotary_dim, rope_layout)
        rope_shape = (max_position if fuse_rope else 1, rope_cols)
        online_softmax = make_online_softmax_with_mask_guard(
            softmax_scale, accum_dtype, block_m, block_n
        )
        apply_softcap = (
            make_apply_softcap(
                1.0 if native_fp8 else score_scale,
                softcap,
                accum_dtype,
                block_m,
                block_n,
            )
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
        apply_transformed_mask = make_bottom_right_attention_mask(
            is_causal,
            window_size_left,
            window_size_right,
            accum_dtype,
            block_m,
            block_n,
            preserve_valid=True,
        )
        rescale = make_rescale(block_m, dim)
        fp8_tile_update = make_native_fp8_prefill_tile_update(
            is_causal=is_causal,
            softcap=softcap,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            accum_dtype=accum_dtype,
            block_m=block_m,
            block_n=block_n,
            dim=dim,
        )

        @T.prim_func
        def _gqa_prefill_varlen_fwd_main(
            q: T.Tensor(q_shape, input_dtype),  # type: ignore
            k: T.Tensor(kv_shape, input_dtype),  # type: ignore
            v: T.Tensor(kv_shape, input_dtype),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
            q_scale: T.Tensor(scale_shape, T.float32),  # type: ignore
            k_scale: T.Tensor(scale_shape, T.float32),  # type: ignore
            v_scale: T.Tensor(scale_shape, T.float32),  # type: ignore
            cos_table: T.Tensor(rope_shape, output_dtype),  # type: ignore
            sin_table: T.Tensor(rope_shape, output_dtype),  # type: ignore
            max_seqlen_q: T.int32,  # type: ignore
            max_seqlen_kv: T.int32,  # type: ignore
            output: T.Tensor(q_shape, output_dtype),  # type: ignore
            lse: T.Tensor([heads, total_q], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(max_seqlen_q, block_m), heads, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                q_shared = T.alloc_shared([block_m, dim], input_dtype)
                k_shared = T.alloc_shared([block_n, dim], input_dtype)
                v_shared = T.alloc_shared([block_n, dim], input_dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                if native_fp8:
                    # The generic TileLang fragment inferred for QK does not
                    # match the RS-PV operand layout. Keep this correctness
                    # schedule explicit until the raw-PTX PV contract is
                    # generalized beyond its fixed BN224 tile.
                    acc_s_cast = T.alloc_shared([block_m, block_n], input_dtype)
                else:
                    acc_s_cast = T.alloc_fragment([block_m, block_n], input_dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)
                inv_logsum = T.alloc_fragment([block_m], accum_dtype)

                q_start = cu_seqlens_q[bz]
                kv_start = cu_seqlens_kv[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                kv_len = cu_seqlens_kv[bz + 1] - kv_start
                causal_offset = kv_len - q_len
                cur_kv_head = by // groups
                score_descale = q_scale[bz, cur_kv_head] * k_scale[bz, cur_kv_head]
                value_descale = v_scale[bz, cur_kv_head]

                if fuse_rope:
                    for i, d in T.Parallel(block_m, dim):
                        q_pos = bx * block_m + i
                        if q_pos < q_len:
                            paired_d = paired_dim(d)
                            value = q[q_start + q_pos, by, d]
                            paired_value = q[q_start + q_pos, by, paired_d]
                            q_shared[i, d] = rotate(
                                value, paired_value, causal_offset + q_pos, d,
                                cos_table, sin_table)
                        else:
                            q_shared[i, d] = T.cast(0, input_dtype)
                elif (bx + 1) * block_m <= q_len:
                    T.copy(
                        q[q_start + bx * block_m : q_start + (bx + 1) * block_m, by, :],
                        q_shared,
                        disable_tma=True,
                    )
                else:
                    for i, d in T.Parallel(block_m, dim):
                        q_pos = bx * block_m + i
                        if q_pos < q_len:
                            q_shared[i, d] = q[q_start + q_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, input_dtype)

                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    k_end = T.ceildiv(T.min(kv_len, causal_offset + (bx + 1) * block_m), block_n)
                elif window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(
                            kv_len,
                            causal_offset + (bx + 1) * block_m + window_size_right,
                        ),
                        block_n,
                    )
                else:
                    k_end = T.ceildiv(kv_len, block_n)

                if window_size_left >= 0:
                    k_start = T.max(0, causal_offset + bx * block_m - window_size_left) // block_n
                else:
                    k_start = 0
                loop_range = T.max(k_end - k_start, 0)

                for k_offset in T.Pipelined(loop_range, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    tile_start = k_idx * block_n
                    tile_end = (k_idx + 1) * block_n
                    if fuse_rope:
                        for j, d in T.Parallel(block_n, dim):
                            kv_pos = tile_start + j
                            if kv_pos < kv_len:
                                paired_d = paired_dim(d)
                                value = k[kv_start + kv_pos, cur_kv_head, d]
                                paired_value = k[kv_start + kv_pos, cur_kv_head, paired_d]
                                k_shared[j, d] = rotate(
                                    value, paired_value, kv_pos, d,
                                    cos_table, sin_table)
                                v_shared[j, d] = v[kv_start + kv_pos, cur_kv_head, d]
                            else:
                                k_shared[j, d] = T.cast(0, input_dtype)
                                v_shared[j, d] = T.cast(0, input_dtype)
                    elif tile_end <= kv_len:
                        T.copy(
                            k[kv_start + tile_start : kv_start + tile_end, cur_kv_head, :],
                            k_shared,
                            disable_tma=True,
                        )
                        T.copy(
                            v[kv_start + tile_start : kv_start + tile_end, cur_kv_head, :],
                            v_shared,
                            disable_tma=True,
                        )
                    else:
                        for j, d in T.Parallel(block_n, dim):
                            kv_pos = tile_start + j
                            if kv_pos < kv_len:
                                k_shared[j, d] = k[kv_start + kv_pos, cur_kv_head, d]
                                v_shared[j, d] = v[kv_start + kv_pos, cur_kv_head, d]
                            else:
                                k_shared[j, d] = T.cast(0, input_dtype)
                                v_shared[j, d] = T.cast(0, input_dtype)

                    if native_fp8:
                        fp8_tile_update(
                            q_shared,
                            k_shared,
                            v_shared,
                            acc_s,
                            acc_s_cast,
                            acc_o,
                            scores_max,
                            scores_max_prev,
                            scores_scale,
                            scores_sum,
                            logsum,
                            k_idx,
                            bx,
                            q_len,
                            kv_len,
                            causal_offset,
                            score_descale * score_scale,
                        )
                    else:
                        if use_softcap:
                            T.clear(acc_s)
                        else:
                            apply_mask(acc_s, k_idx, bx, q_len, kv_len, causal_offset)
                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        if use_softcap:
                            apply_softcap(acc_s)
                            apply_transformed_mask(acc_s, k_idx, bx, q_len, kv_len, causal_offset)
                        online_softmax(
                            acc_s,
                            scores_max,
                            scores_max_prev,
                            scores_scale,
                            scores_sum,
                            logsum,
                        )
                        T.copy(acc_s, acc_s_cast)
                        rescale(acc_o, scores_scale)
                        T.gemm(
                            acc_s_cast,
                            v_shared,
                            acc_o,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                if (bx + 1) * block_m <= q_len:
                    for i in T.Parallel(block_m):
                        inv_logsum[i] = T.cast(1, accum_dtype) / logsum[i]
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= inv_logsum[i] * (value_descale if native_fp8 else 1.0)
                    T.copy(
                        acc_o,
                        output[q_start + bx * block_m : q_start + (bx + 1) * block_m, by, :],
                        disable_tma=True,
                    )
                    for i in T.Parallel(block_m):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * softmax_scale
                    T.copy(
                        logsum,
                        lse[by, q_start + bx * block_m : q_start + (bx + 1) * block_m],
                        disable_tma=True,
                    )
                else:
                    for i in T.Parallel(block_m):
                        q_pos = bx * block_m + i
                        if q_pos < q_len:
                            inv_logsum[i] = T.cast(1, accum_dtype) / logsum[i]
                    for i, j in T.Parallel(block_m, dim):
                        q_pos = bx * block_m + i
                        if q_pos < q_len:
                            output[q_start + q_pos, by, j] = (
                                acc_o[i, j] * inv_logsum[i] * (value_descale if native_fp8 else 1.0)
                            )
                    for i in T.Parallel(block_m):
                        q_pos = bx * block_m + i
                        if q_pos < q_len:
                            lse[by, q_start + q_pos] = (
                                T.log2(logsum[i]) + scores_max[i] * softmax_scale
                            )

        return _gqa_prefill_varlen_fwd_main

    return _gqa_prefill_varlen_fwd_func


@torch.library.custom_op("top::gqa_prefill_varlen_fwd_wrapped_kernel", mutates_args=())
def _gqa_prefill_varlen_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    input_dtype: str,
    output_dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_prefill_varlen_fwd_kernel(
        batch,
        heads,
        heads_kv,
        total_q,
        total_kv,
        dim,
        is_causal,
        sm_scale,
        softcap,
        window_size_left,
        window_size_right,
        input_dtype,
        output_dtype,
        fuse_rope,
        max_position,
        rotary_dim,
        rope_layout,
    )(block_m, block_n, num_stages, threads)(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        q_scale,
        k_scale,
        v_scale,
        rope_cos,
        rope_sin,
        max_seqlen_q,
        max_seqlen_kv,
    )


@_gqa_prefill_varlen_fwd_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    input_dtype: str,
    output_dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float16 if output_dtype == "float16" else torch.bfloat16
    fake_o = torch.empty([total_q, heads, dim], dtype=dtype, device=q.device)
    fake_lse = torch.empty([heads, total_q], dtype=torch.float32, device=q.device)
    return fake_o, fake_lse


class GQAPrefillVarlenFwdKernel(PackedPrefillKernel):
    """Ragged packed prefill: per-request ranges of unequal length.

    This is a Varlen-Op implementation. The Op fixes the packed/ragged
    topology; this predicate only distinguishes semantic kernel families.
    """

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True

    @classmethod
    def applies(cls, call) -> bool:
        return not call.is_fp8 and not call.is_uniform

    def _build_program(self) -> None:
        # The program is specialized on the packed totals, which are known per
        # call, so there is nothing to build until forward runs.
        self.kernel = None

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
        configs = list(itertools.product([32, 64, 128], [32, 64, 128], [1, 2, 3], [128, 256]))
        return [
            {"block_m": c[0], "block_n": c[1], "num_stages": c[2], "threads": c[3]} for c in configs
        ]

    @property
    def input_dtype_str(self) -> str:
        return self.dtype_str

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
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q_scale is None or k_scale is None or v_scale is None:
            raise ValueError("packed prefill requires resolved q/k/v scale tensors")
        if rope_cos is None or rope_sin is None:
            raise ValueError("packed prefill requires prepared RoPE or dummy tables")
        total_q, total_kv = q.shape[0], k.shape[0]
        output, _ = _gqa_prefill_varlen_fwd_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            total_q,
            total_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.window_size_left,
            self.window_size_right,
            self.input_dtype_str,
            self.dtype_str,
            self.fuse_rope,
            self.max_position or 1,
            self.rotary_dim or 0,
            self.rope_layout,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
            self.max_seqlen_q,
            self.max_seqlen_kv,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            q_scale,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
        )
        return output


class GQAPrefillVarlenFP8TensorCoreFwdKernel(GQAPrefillVarlenFwdKernel):
    """SM90 ragged prefill using native FP8 Tensor Core QK and PV."""

    supported_archs: list[int] = [90]
    general: bool = True

    @classmethod
    def applies(cls, call) -> bool:
        return call.is_fp8 and not call.is_uniform and call.dim == 128

    @property
    def input_dtype_str(self) -> str:
        return "float8_e4m3fn"
