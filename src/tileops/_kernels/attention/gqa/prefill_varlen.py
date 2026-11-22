"""Packed variable-length GQA prefill forward kernel.

Inputs use THD layout:
  q: [T_q, H, D]
  k/v: [T_kv, H_kv, D]

``cu_seqlens_q`` and ``cu_seqlens_kv`` describe per-request packed ranges.
Causal masking and the sliding window use bottom-right alignment per request,
matching the dense prefill contract when q_len may be smaller than kv_len.
"""

import itertools
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.online_softmax import (
    make_apply_softcap,
    make_online_softmax_with_mask_guard,
    make_rescale,
    make_sink_scale,
)
from tileops._kernels.attention.varlen import VarlenKernel
from tileops._kernels.attention.varlen_rope import make_varlen_query_rope
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import (
    LOG2E,
    SHARED_BUFFER_ALIGN_BYTES,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops._kernels.grouped_tiling import GroupTiling

__all__ = ["GQAPrefillVarlenFwdKernel"]


def stages_score_tile(block_m: int, threads: int) -> bool:
    """Whether the score tile goes through shared memory: TileLang finds no register
    layout for it when a warpgroup holds fewer than ``WGMMA_ROWS`` rows."""
    warpgroups = threads // WARPGROUP_THREADS
    return block_m // warpgroups < WGMMA_ROWS


def varlen_key_tiles(q_row, rows, kv_len, offset, block_n, is_causal, window_left, window_right):
    """The first key tile a query tile of *rows* rows from position *q_row* scans, and how many.

    A request's queries are aligned to the end of its keys by *offset*; key tiles wholly left
    of the window are skipped, not masked.
    """
    if is_causal:
        stop = T.max(0, T.ceildiv(T.min(kv_len, offset + q_row + rows), block_n))
    elif window_right >= 0:
        stop = T.max(0, T.ceildiv(T.min(kv_len, offset + q_row + rows + window_right), block_n))
    else:
        stop = T.ceildiv(kv_len, block_n)
    if window_left < 0:
        return 0, stop
    first = T.max(0, offset + q_row - window_left) // block_n
    return first, T.max(0, stop - first)


def varlen_visible(q_pos, kv_pos, q_len, kv_len, offset, is_causal, window_left, window_right):
    """Whether query *q_pos* of a request sees its key *kv_pos* under the mask and window."""
    valid = (q_pos < q_len) & (kv_pos < kv_len)
    if is_causal:
        valid = valid & (kv_pos <= q_pos + offset)
    elif window_right >= 0:
        valid = valid & (kv_pos <= q_pos + offset + window_right)
    if window_left >= 0:
        valid = valid & (kv_pos >= q_pos + offset - window_left)
    return valid


def _aligned_bytes(buffers: list[int]) -> int:
    """Shared memory *buffers* take, each rounded up to TileLang's allocation alignment."""
    align = SHARED_BUFFER_ALIGN_BYTES
    return sum(-(-size // align) * align for size in buffers)


def _narrow_config(dim: int) -> dict:
    """The one-stage, one-warpgroup plan: 64 query rows, and 32 keys past 128 channels."""
    return {"block_m": 64, "block_n": 64 if dim <= 128 else 32, "num_stages": 1, "threads": 128}


@target_cache(maxsize=32)
def _gqa_prefill_varlen_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float | None = None,
    softcap: float = 0.0,
    dtype: str = "float16",
    window_size_left: int = -1,
    window_size_right: int = -1,
    fuse_rope: bool = False,
    max_position: int = 1,
    rotary_dim: int = 0,
    rope_layout: str = "neox",
    rope_dtype: str = "",
    has_sinks: bool = False,
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap or score_scale <= 0 else score_scale * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[7] if fuse_rope else [5],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
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
        p_via_shared = stages_score_tile(block_m, threads)
        q_tiling = GroupTiling(batch, block_m)
        num_q_tiles = q_tiling.tile_upper_bound(total_q)
        rotate_query_tile = (
            make_varlen_query_rope(
                block_m,
                rotary_dim,
                rope_layout,
                max_position,
                dtype,
                threads=threads,
                split_reads=True,
            )
            if fuse_rope
            else None
        )

        sink_scale = make_sink_scale(scale, accum_dtype, block_m)

        def normalized(acc_o, inv_logsum, i, j):
            """Output element (i, j); a row the sinks zero stays zero whatever acc_o holds."""
            if has_sinks:
                return T.if_then_else(inv_logsum[i] > 0, acc_o[i, j] * inv_logsum[i], 0)
            return acc_o[i, j] * inv_logsum[i]

        @T.macro
        def attend(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, output, sinks, q_tile, by,
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

                k_first, loop_range = varlen_key_tiles(
                    q_row[0], block_m, kv_len, causal_offset, block_n,
                    is_causal, window_size_left, window_size_right,
                )  # fmt: skip
                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    tile_start = (k_first + k_idx) * block_n
                    tile_end = tile_start + block_n
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
                        valid = varlen_visible(
                            q_pos, kv_pos, q_len, kv_len, causal_offset,
                            is_causal, window_size_left, window_size_right,
                        )  # fmt: skip
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
                    elif score_scale <= 0:
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.if_then_else(
                                acc_s[i, j] == -T.infinity(accum_dtype),
                                -T.infinity(accum_dtype),
                                acc_s[i, j] * score_scale,
                            )
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

                for i in T.Parallel(block_m):
                    inv_logsum[i] = T.if_then_else(logsum[i] > 0, 1.0 / logsum[i], 0)
                if has_sinks:
                    sink_scale(scores_max, logsum, sinks, by, inv_logsum)
                if q_row[0] + block_m <= q_len:
                    # The query tile is dead after the last key tile, so the output
                    # goes through its shared buffer and leaves as whole rows.
                    for i, j in T.Parallel(block_m, dim):
                        q_shared[i, j] = normalized(acc_o, inv_logsum, i, j)
                    T.copy(
                        q_shared,
                        output[q_start + q_row[0] : q_start + q_row[0] + block_m, by, :],
                        disable_tma=True,
                    )
                else:
                    for i, j in T.Parallel(block_m, dim):
                        q_pos = q_row[0] + i
                        if q_pos < q_len:
                            output[q_start + q_pos, by, j] = normalized(acc_o, inv_logsum, i, j)

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
                sinks: T.Tensor([heads], "float32"),  # type: ignore
            ) -> None:
                with T.Kernel(num_q_tiles, heads, threads=threads) as (q_tile, by):
                    attend(
                        q, k, v, cu_seqlens_q, cu_seqlens_kv, output, sinks, q_tile, by,
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
            sinks: T.Tensor([heads], "float32"),  # type: ignore
        ) -> None:
            with T.Kernel(num_q_tiles, heads, threads=threads) as (q_tile, by):
                attend(q, k, v, cu_seqlens_q, cu_seqlens_kv, output, sinks, q_tile, by)

        return _gqa_prefill_varlen_fwd_main

    return _gqa_prefill_varlen_fwd_func


def prefill_shared_bytes(config: dict, *, batch: int, dim: int, element: int) -> int:
    """Count the one-stage program's shared buffers with allocation alignment."""
    block_m, block_n = config["block_m"], config["block_n"]
    tile = block_n * dim * element
    buffers = [block_m * dim * element, tile, tile, 4 * (batch + 1)]
    if stages_score_tile(block_m, config["threads"]):
        buffers += [block_m * block_n * element, 4 * config["threads"], 4 * config["threads"]]
    return _aligned_bytes(buffers)


def prefill_default_config(*, batch: int, dim: int, element: int, budget: int):
    """Select the existing one-stage plan, or report that no candidate fits."""
    narrow = _narrow_config(dim)
    candidates = [narrow, {**narrow, "block_m": 32}, {**narrow, "block_m": 32, "block_n": 16}]
    if 256 < dim <= 512:
        candidates.insert(0, {"block_m": 64, "block_n": 64, "num_stages": 1, "threads": 256})
    return next(
        (
            config
            for config in candidates
            if prefill_shared_bytes(config, batch=batch, dim=dim, element=element) <= budget
        ),
        None,
    )


class GQAPrefillVarlenFwdKernel(VarlenKernel):
    """Packed prefill over per-request ranges of any length, with or without a sliding window."""

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True

    @classmethod
    def refusal(cls, call) -> "str | None":
        if call.is_fp8:
            return "does not serve FP8"
        # The score GEMM contracts the head dim in MMA steps of 16.
        if call.dim % 16:
            return f"requires a head dim that is a multiple of 16, got {call.dim}"
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
            self.has_sinks,
        )

    @property
    def default_config(self) -> dict:
        """The first candidate the device's shared memory holds."""
        plan = prefill_default_config(
            batch=self.batch,
            dim=self.dim,
            element=self.dtype.itemsize,
            budget=self.build_target.smem_budget,
        )
        return plan or _narrow_config(self.dim)

    def tuning_candidates(self, configs, untuned=None) -> list[dict]:
        candidates = super().tuning_candidates(configs, untuned)
        budget = self.build_target.smem_budget
        # Q and K are simultaneously live at the score product. This lower bound
        # rejects impossible tiles without assuming pipeline allocation or aliasing.
        return [
            config
            for config in candidates
            if (config["block_m"] + config["block_n"]) * self.dim * self.dtype.itemsize <= budget
        ]

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
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        program = self.program(self._compile)
        if self.key_rope is None:
            return program(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                sinks if self.has_sinks else self._unused_sinks,
            )
        k_rot = self.key_rope(k, cu_seqlens_kv, rope_cos, rope_sin)
        return program(
            q,
            k_rot,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            rope_cos,
            rope_sin,
            sinks if self.has_sinks else self._unused_sinks,
        )

    def _compile(self, cfg: dict):
        """The program for *cfg*."""
        return self.kernel(cfg["block_m"], cfg["block_n"], cfg["num_stages"], cfg["threads"])
