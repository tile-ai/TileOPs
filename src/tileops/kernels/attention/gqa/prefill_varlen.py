"""Packed variable-length GQA prefill forward kernel.

Inputs use THD layout:
  q: [T_q, H, D]
  k/v: [T_kv, H_kv, D]

``cu_seqlens_q`` and ``cu_seqlens_kv`` describe per-request packed ranges.
Causal masking and the sliding window use bottom-right alignment per request,
matching the dense prefill contract when q_len may be smaller than kv_len.
"""

import functools
import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.online_softmax import (
    make_apply_softcap,
    make_online_softmax_with_mask_guard,
    make_rescale,
)
from tileops.kernels.attention.varlen import VarlenKernel
from tileops.kernels.attention.varlen_rope import make_varlen_query_rope
from tileops.kernels.constants import (
    LOG2E,
    SHARED_BUFFER_ALIGN_BYTES,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.utils import get_shared_memory_optin

__all__ = ["GQAPrefillVarlenFwdKernel"]


def _stages_score_tile(block_m: int, threads: int) -> bool:
    """Whether the score tile goes through shared memory: TileLang finds no register
    layout for it when a warpgroup holds fewer than ``WGMMA_ROWS`` rows."""
    warpgroups = threads // WARPGROUP_THREADS
    return block_m // warpgroups < WGMMA_ROWS


@functools.lru_cache(maxsize=32)
def _gqa_prefill_varlen_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: float = 0.0,
    dtype: str = "float16",
    window_size_left: int = -1,
    window_size_right: int = -1,
    fuse_rope: bool = False,
    max_position: int = 1,
    rotary_dim: int = 0,
    rope_layout: str = "neox",
    rope_dtype: str = "",
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    has_left = window_size_left >= 0
    has_right = window_size_right >= 0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[7] if fuse_rope else [5],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_prefill_varlen_fwd_func(
        block_m: int, block_n: int, num_stages: int, threads: int
    ) -> Callable:
        total_q = T.dynamic("total_q")
        total_kv = T.dynamic("total_kv")
        q_shape = (total_q, heads, dim)
        kv_shape = (total_kv, heads_kv, dim)
        online_softmax = make_online_softmax_with_mask_guard(scale, accum_dtype, block_m, block_n)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_m, dim)
        p_via_shared = _stages_score_tile(block_m, threads)
        q_tiling = GroupTiling(batch, block_m)
        num_q_tiles = q_tiling.tile_upper_bound(total_q)
        rotate_query_tile = (
            make_varlen_query_rope(
                block_m, rotary_dim, rope_layout, max_position, dtype, rope_dtype
            )
            if fuse_rope
            else None
        )

        @T.macro
        def attend(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, output, q_tile, by,
            rope_cos=None, rope_sin=None,
        ):  # fmt: skip
            """The scan, with the query tile rotated in place when the tables are passed."""
            q_shared = T.alloc_shared([block_m, dim], dtype)
            k_shared = T.alloc_shared([block_n, dim], dtype)
            v_shared = T.alloc_shared([block_n, dim], dtype)
            tile_cum = T.alloc_shared([batch + 1], "int32")
            acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
            if p_via_shared:
                acc_s_cast = T.alloc_shared([block_m, block_n], dtype)
            else:
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_m], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
            scores_scale = T.alloc_fragment([block_m], accum_dtype)
            scores_sum = T.alloc_fragment([block_m], accum_dtype)
            logsum = T.alloc_fragment([block_m], accum_dtype)
            inv_logsum = T.alloc_fragment([block_m], accum_dtype)
            lo = T.alloc_local([1], "int32")
            hi = T.alloc_local([1], "int32")
            q_row = T.alloc_local([1], "int32")
            request = T.alloc_local([1], "int32")

            q_tiling.cumsum_offsets(cu_seqlens_q, tile_cum)
            if q_tile < tile_cum[batch]:
                q_tiling.decode(q_tile, tile_cum, lo, hi, request, q_row)

                q_start = cu_seqlens_q[request[0]]
                kv_start = cu_seqlens_kv[request[0]]
                q_len = cu_seqlens_q[request[0] + 1] - q_start
                kv_len = cu_seqlens_kv[request[0] + 1] - kv_start
                causal_offset = kv_len - q_len
                cur_kv_head = by // groups

                if q_row[0] + block_m <= q_len:
                    T.copy(
                        q[q_start + q_row[0] : q_start + q_row[0] + block_m, by, :],
                        q_shared,
                        disable_tma=True,
                    )
                else:
                    for i, d in T.Parallel(block_m, dim):
                        q_pos = q_row[0] + i
                        if q_pos < q_len:
                            q_shared[i, d] = q[q_start + q_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, dtype)
                if rope_cos is not None:
                    # Query token i of a request sits at position kv_len - q_len + i.
                    rotate_query_tile(q_shared, rope_cos, rope_sin, causal_offset + q_row[0])

                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    loop_range = T.max(
                        0,
                        T.ceildiv(T.min(kv_len, causal_offset + q_row[0] + block_m), block_n),
                    )
                elif has_right:
                    loop_range = T.max(
                        0,
                        T.ceildiv(
                            T.min(
                                kv_len,
                                causal_offset + q_row[0] + block_m + window_size_right,
                            ),
                            block_n,
                        ),
                    )
                else:
                    loop_range = T.ceildiv(kv_len, block_n)
                # Key tiles wholly left of the window are skipped, not masked.
                if has_left:
                    k_first = T.max(0, causal_offset + q_row[0] - window_size_left) // block_n
                    loop_range = T.max(0, loop_range - k_first)

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    if has_left:
                        tile_start = (k_first + k_idx) * block_n
                        tile_end = (k_first + k_idx + 1) * block_n
                    else:
                        tile_start = k_idx * block_n
                        tile_end = (k_idx + 1) * block_n
                    if tile_end <= kv_len:
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
                                k_shared[j, d] = T.cast(0, dtype)
                                v_shared[j, d] = T.cast(0, dtype)

                    for i, j in T.Parallel(block_m, block_n):
                        q_pos = q_row[0] + i
                        kv_pos = tile_start + j
                        if is_causal:
                            valid = (
                                (q_pos < q_len)
                                & (kv_pos < kv_len)
                                & (kv_pos <= q_pos + causal_offset)
                            )
                        elif has_right:
                            valid = (
                                (q_pos < q_len)
                                & (kv_pos < kv_len)
                                & (kv_pos <= q_pos + causal_offset + window_size_right)
                            )
                        else:
                            valid = (q_pos < q_len) & (kv_pos < kv_len)
                        if has_left:
                            valid = valid & (kv_pos >= q_pos + causal_offset - window_size_left)
                        acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    if use_softcap:
                        apply_softcap(acc_s)
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
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                if q_row[0] + block_m <= q_len:
                    for i in T.Parallel(block_m):
                        inv_logsum[i] = T.if_then_else(
                            logsum[i] > 0,
                            T.cast(1, accum_dtype) / logsum[i],
                            T.cast(0, accum_dtype),
                        )
                    # The query tile is dead after the last key tile, so the output
                    # goes through its shared buffer and leaves as whole rows.
                    for i, j in T.Parallel(block_m, dim):
                        q_shared[i, j] = acc_o[i, j] * inv_logsum[i]
                    T.copy(
                        q_shared,
                        output[q_start + q_row[0] : q_start + q_row[0] + block_m, by, :],
                        disable_tma=True,
                    )
                else:
                    for i in T.Parallel(block_m):
                        q_pos = q_row[0] + i
                        if q_pos < q_len:
                            inv_logsum[i] = T.if_then_else(
                                logsum[i] > 0,
                                T.cast(1, accum_dtype) / logsum[i],
                                T.cast(0, accum_dtype),
                            )
                    for i, j in T.Parallel(block_m, dim):
                        q_pos = q_row[0] + i
                        if q_pos < q_len:
                            output[q_start + q_pos, by, j] = acc_o[i, j] * inv_logsum[i]

        if fuse_rope:
            half = rotary_dim // 2
            table_dtype = rope_dtype or dtype

            @T.prim_func
            def _gqa_prefill_varlen_rope_fwd_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k: T.Tensor(kv_shape, dtype),  # type: ignore
                v: T.Tensor(kv_shape, dtype),  # type: ignore
                cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
                cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
                rope_cos: T.Tensor([max_position, half], table_dtype),  # type: ignore
                rope_sin: T.Tensor([max_position, half], table_dtype),  # type: ignore
                output: T.Tensor(q_shape, dtype),  # type: ignore
            ) -> None:
                with T.Kernel(num_q_tiles, heads, threads=threads) as (q_tile, by):
                    attend(
                        q, k, v, cu_seqlens_q, cu_seqlens_kv, output, q_tile, by,
                        rope_cos, rope_sin,
                    )  # fmt: skip

            return _gqa_prefill_varlen_rope_fwd_main

        @T.prim_func
        def _gqa_prefill_varlen_fwd_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
            output: T.Tensor(q_shape, dtype),  # type: ignore
        ) -> None:
            with T.Kernel(num_q_tiles, heads, threads=threads) as (q_tile, by):
                attend(q, k, v, cu_seqlens_q, cu_seqlens_kv, output, q_tile, by)

        return _gqa_prefill_varlen_fwd_main

    return _gqa_prefill_varlen_fwd_func


class GQAPrefillVarlenFwdKernel(VarlenKernel):
    """Packed prefill over per-request ranges of any length, with or without a sliding window."""

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True

    @classmethod
    def refusal(cls, call) -> "str | None":
        if call.is_fp8:
            return "does not serve FP8"
        return super().refusal(call)

    def _make_kernel(self) -> Callable:
        return _gqa_prefill_varlen_fwd_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self.window_size_left,
            self.window_size_right,
            self.fuse_rope,
            self.max_position,
            self.rotary_dim,
            self.rope_layout,
            self.rope_table_dtype_str,
        )

    @property
    def default_config(self) -> dict:
        """The first candidate the device's shared memory holds."""
        narrow = {
            "block_m": 64,
            "block_n": 64 if self.dim <= 128 else 32,
            "num_stages": 1,
            "threads": 128,
        }
        candidates = [narrow, {**narrow, "block_m": 32}, {**narrow, "block_m": 32, "block_n": 16}]
        if 256 < self.dim <= 512:
            candidates.insert(0, {"block_m": 64, "block_n": 64, "num_stages": 1, "threads": 256})
        cap = get_shared_memory_optin(self.device_index)
        return next((c for c in candidates if self._shared_bytes(c) <= cap), narrow)

    def _shared_bytes(self, config: dict) -> int:
        """Shared memory a one-stage *config* allocates, buffer by buffer as TileLang does."""
        elem = self.dtype.itemsize
        block_m, block_n = config["block_m"], config["block_n"]
        tile = block_n * self.dim * elem
        buffers = [block_m * self.dim * elem, tile, tile, 4 * (self.batch + 1)]
        if _stages_score_tile(block_m, config["threads"]):
            buffers += [block_m * block_n * elem, 4 * config["threads"], 4 * config["threads"]]
        align = SHARED_BUFFER_ALIGN_BYTES
        return sum(-(-b // align) * align for b in buffers)

    @property
    def autotune_configs(self) -> list[dict]:
        configs = list(itertools.product([32, 64, 128], [32, 64, 128], [1, 2, 3], [128, 256]))
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
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        program = self.kernel(
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
        )
        if self.key_rope is None:
            return program(q, k, v, cu_seqlens_q, cu_seqlens_kv)
        k_rot = self.key_rope(k, cu_seqlens_kv, rope_cos, rope_sin)
        return program(q, k_rot, v, cu_seqlens_q, cu_seqlens_kv, rope_cos, rope_sin)
