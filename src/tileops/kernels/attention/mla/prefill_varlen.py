"""Packed variable-length MLA prefill forward kernel.

Inputs use THD layout, after the latent is decompressed::

  q:      [T_q, H, DN + PE]   the nope half then the rope half
  k_nope: [T_q, H, DN]
  k_pe:   [T_q, PE]           one row per token, shared by every head
  v:      [T_q, H, DV]

The key of head ``h`` is ``k_nope[:, h] | k_pe``, so the score is the sum of a
``DN``-wide contraction against ``k_nope`` and a ``PE``-wide one against
``k_pe``.  Both land in one accumulator, which is what lets ``k_pe`` stay one
row per token: building the concatenated ``DN + PE`` key instead would carry the
rope half through memory once per head rather than once per token.

Queries and keys are the same tokens, so the mask is causal on the diagonal and
one ``cu_seqlens`` describes both.  ``lse`` is returned in float32 so a caller
merging chunked-context partials can combine them.
"""

import functools
import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import MlaVarlenCall, MlaVarlenFwdInterface
from tileops.kernels.attention.online_softmax import (
    make_online_softmax_with_mask_guard,
    make_rescale,
)
from tileops.kernels.constants import (
    LOG2E,
    SHARED_BUFFER_ALIGN_BYTES,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin

__all__ = ["MLAVarlenPrefillFwdKernel"]

_CANDIDATES = (
    {"block_m": 128, "block_n": 128, "num_stages": 2, "threads": 256},
    {"block_m": 128, "block_n": 64, "num_stages": 2, "threads": 256},
    {"block_m": 64, "block_n": 64, "num_stages": 2, "threads": 128},
)
_NARROW = _CANDIDATES[-1]


def _stages_score_tile(block_m: int, threads: int) -> bool:
    """Whether the score tile goes through shared memory.

    TileLang finds no register layout for it when a warpgroup holds fewer than
    ``WGMMA_ROWS`` rows, and staging it also takes it out of the register budget
    that is what caps this kernel at one block per SM.
    """
    warpgroups = threads // WARPGROUP_THREADS
    return warpgroups > 1 and block_m // warpgroups < WGMMA_ROWS


@functools.lru_cache(maxsize=32)
def _mla_varlen_fwd_kernel(
    batch: int,
    heads: int,
    dim_nope: int,
    dim_pe: int,
    dim_v: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    dtype: str = "bfloat16",
) -> Callable:
    """Build the dynamic TileLang program factory for one call shape."""
    score_scale = (dim_nope + dim_pe) ** -0.5 if sm_scale is None else sm_scale
    scale = score_scale * LOG2E
    accum_dtype = "float"
    dim_qk = dim_nope + dim_pe

    @tilelang.jit(
        out_idx=[5, 6],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _mla_varlen_fwd_func(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:
        total_q = T.dynamic("total_q")
        q_shape = (total_q, heads, dim_qk)
        k_nope_shape = (total_q, heads, dim_nope)
        k_pe_shape = (total_q, dim_pe)
        v_shape = (total_q, heads, dim_v)
        out_shape = (total_q, heads, dim_v)
        lse_shape = (total_q, heads)
        online_softmax = make_online_softmax_with_mask_guard(scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim_v)
        p_via_shared = _stages_score_tile(block_m, threads)
        q_tiling = GroupTiling(batch, block_m)
        num_q_tiles = q_tiling.tile_upper_bound(total_q)

        @T.prim_func
        def _mla_varlen_fwd_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k_nope: T.Tensor(k_nope_shape, dtype),  # type: ignore
            k_pe: T.Tensor(k_pe_shape, dtype),  # type: ignore
            v: T.Tensor(v_shape, dtype),  # type: ignore
            cu_seqlens: T.Tensor([batch + 1], T.int32),  # type: ignore
            output: T.Tensor(out_shape, dtype),  # type: ignore
            lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(num_q_tiles, heads, threads=threads) as (q_tile, by):
                q_nope_shared = T.alloc_shared([block_m, dim_nope], dtype)
                q_pe_shared = T.alloc_shared([block_m, dim_pe], dtype)
                k_nope_shared = T.alloc_shared([block_n, dim_nope], dtype)
                k_pe_shared = T.alloc_shared([block_n, dim_pe], dtype)
                v_shared = T.alloc_shared([block_n, dim_v], dtype)
                tile_cum = T.alloc_shared([batch + 1], "int32")
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                if p_via_shared:
                    acc_s_cast = T.alloc_shared([block_m, block_n], dtype)
                else:
                    acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim_v], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)
                inv_logsum = T.alloc_fragment([block_m], accum_dtype)
                row_lse = T.alloc_fragment([block_m], accum_dtype)
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                q_row = T.alloc_local([1], "int32")
                request = T.alloc_local([1], "int32")

                q_tiling.cumsum_offsets(cu_seqlens, tile_cum)
                if q_tile < tile_cum[batch]:
                    q_tiling.decode(q_tile, tile_cum, lo, hi, request, q_row)

                    seq_start = cu_seqlens[request[0]]
                    seq_len = cu_seqlens[request[0] + 1] - seq_start

                    # q is one packed row of DN + PE; the two halves are staged
                    # apart because they contract against different keys.
                    for i, d in T.Parallel(block_m, dim_nope):
                        q_pos = q_row[0] + i
                        if q_pos < seq_len:
                            q_nope_shared[i, d] = q[seq_start + q_pos, by, d]
                        else:
                            q_nope_shared[i, d] = T.cast(0, dtype)
                    for i, d in T.Parallel(block_m, dim_pe):
                        q_pos = q_row[0] + i
                        if q_pos < seq_len:
                            q_pe_shared[i, d] = q[seq_start + q_pos, by, dim_nope + d]
                        else:
                            q_pe_shared[i, d] = T.cast(0, dtype)

                    T.clear(acc_o)
                    T.clear(logsum)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    # Self-attention on the diagonal: a causal row reads no key
                    # past its own position, so the scan stops at this tile's
                    # last query row.
                    loop_range = (
                        T.max(0, T.ceildiv(T.min(seq_len, q_row[0] + block_m), block_n))
                        if is_causal
                        else T.ceildiv(seq_len, block_n)
                    )

                    for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                        tile_start = k_idx * block_n
                        tile_end = (k_idx + 1) * block_n
                        if tile_end <= seq_len:
                            T.copy(
                                k_nope[seq_start + tile_start : seq_start + tile_end, by, :],
                                k_nope_shared,
                                disable_tma=True,
                            )
                            # read once per token, not once per head
                            T.copy(
                                k_pe[seq_start + tile_start : seq_start + tile_end, :],
                                k_pe_shared,
                                disable_tma=True,
                            )
                            T.copy(
                                v[seq_start + tile_start : seq_start + tile_end, by, :],
                                v_shared,
                                disable_tma=True,
                            )
                        else:
                            for j, d in T.Parallel(block_n, dim_nope):
                                kv_pos = tile_start + j
                                if kv_pos < seq_len:
                                    k_nope_shared[j, d] = k_nope[seq_start + kv_pos, by, d]
                                else:
                                    k_nope_shared[j, d] = T.cast(0, dtype)
                            for j, d in T.Parallel(block_n, dim_pe):
                                kv_pos = tile_start + j
                                if kv_pos < seq_len:
                                    k_pe_shared[j, d] = k_pe[seq_start + kv_pos, d]
                                else:
                                    k_pe_shared[j, d] = T.cast(0, dtype)
                            for j, d in T.Parallel(block_n, dim_v):
                                kv_pos = tile_start + j
                                if kv_pos < seq_len:
                                    v_shared[j, d] = v[seq_start + kv_pos, by, d]
                                else:
                                    v_shared[j, d] = T.cast(0, dtype)

                        # A block whose last key is at or before this tile's
                        # first query row is visible to every row of the tile,
                        # and the diagonal is one block wide, so most of the
                        # scan needs no mask and no per-element bounds test.
                        if tile_end <= T.min(q_row[0] + 1, seq_len):
                            T.clear(acc_s)
                        else:
                            for i, j in T.Parallel(block_m, block_n):
                                q_pos = q_row[0] + i
                                kv_pos = tile_start + j
                                if is_causal:
                                    valid = (
                                        (q_pos < seq_len) & (kv_pos < seq_len) & (kv_pos <= q_pos)
                                    )
                                else:
                                    valid = (q_pos < seq_len) & (kv_pos < seq_len)
                                acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))

                        # The two halves of the key, accumulated in one tile.
                        T.gemm(
                            q_nope_shared,
                            k_nope_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            q_pe_shared,
                            k_pe_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
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
                        inv_logsum[i] = T.if_then_else(
                            logsum[i] > 0,
                            T.cast(1, accum_dtype) / logsum[i],
                            T.cast(0, accum_dtype),
                        )
                    for i, j in T.Parallel(block_m, dim_v):
                        acc_o[i, j] *= inv_logsum[i]
                    # log-sum-exp in the natural base, so a caller merging
                    # chunked-context partials needs nothing of this kernel's
                    # exp2 scaling.
                    for i in T.Parallel(block_m):
                        row_lse[i] = T.if_then_else(
                            logsum[i] > 0,
                            T.log(logsum[i]) + scores_max[i] * score_scale,
                            -T.infinity(accum_dtype),
                        )

                    row0 = seq_start + q_row[0]
                    if q_row[0] + block_m <= seq_len:
                        T.copy(acc_o, output[row0 : row0 + block_m, by, :], disable_tma=True)
                        T.copy(row_lse, lse[row0 : row0 + block_m, by])
                    else:
                        for i, j in T.Parallel(block_m, dim_v):
                            if q_row[0] + i < seq_len:
                                output[row0 + i, by, j] = T.cast(acc_o[i, j], dtype)
                        for i in T.Parallel(block_m):
                            if q_row[0] + i < seq_len:
                                lse[row0 + i, by] = row_lse[i]

        return _mla_varlen_fwd_main

    return _mla_varlen_fwd_func


class MLAVarlenPrefillFwdKernel(Kernel, MlaVarlenFwdInterface):
    """Packed-varlen MLA prefill: one query block of one head per CTA."""

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True

    @classmethod
    def applies(cls, call: MlaVarlenCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: MlaVarlenCall) -> Optional[str]:
        """Why *call* is outside the shapes this schedule serves."""
        if call.dim_nope % 16 != 0 or call.dim_pe % 16 != 0 or call.dim_v % 16 != 0:
            return (
                "every head dimension is contracted in 16-wide steps, got "
                f"{call.dim_nope}, {call.dim_pe}, {call.dim_v}"
            )
        need = cls._shared_bytes(
            call.batch, call.dim_nope, call.dim_pe, call.dim_v, call.dtype.itemsize, _NARROW
        )
        if call.smem_budget and need > call.smem_budget:
            return (
                f"needs {need} bytes of shared memory per block at its narrowest tile, over "
                f"the {call.smem_budget} bytes the device allows"
            )
        return None

    @classmethod
    def entry_for(cls, call: MlaVarlenCall) -> Entry:
        args = dict(
            batch=call.batch,
            heads=call.heads,
            dim_nope=call.dim_nope,
            dim_pe=call.dim_pe,
            dim_v=call.dim_v,
            is_causal=call.is_causal,
            sm_scale=call.sm_scale,
            dtype=call.dtype,
            device_index=call.device.index if call.device is not None else None,
        )
        return tuple(args.items()), lambda: cls(**args)

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_nope: int,
        dim_pe: int,
        dim_v: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        """Build the program factory for one call shape.

        Args:
            batch: Requests the packing carries, so the length of ``cu_seqlens`` less one.
            heads: Query heads.
            dim_nope: Width of the key's per-head half.
            dim_pe: Width of the key's shared rope half.
            dim_v: Width of a value row.
            is_causal: Whether a query row reads only keys at or before its position.
            dtype: Input and output dtype.
            sm_scale: Score scale, or ``None`` for ``(dim_nope + dim_pe) ** -0.5``.
            config: Tile sizes and pipeline depth, or ``None`` for the default.
            tune: Whether to autotune when the kernel is first built.
            device_index: CUDA device the program is built for.
        """
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.dim_nope = dim_nope
        self.dim_pe = dim_pe
        self.dim_v = dim_v
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = sm_scale
        self.kernel = _mla_varlen_fwd_kernel(
            batch,
            heads,
            dim_nope,
            dim_pe,
            dim_v,
            is_causal,
            sm_scale,
            self.dtype_str,
        )
        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    def _make_supply_prog(self) -> Callable:
        """Supply valid packed tensors and offsets while autotuning.

        Autotuning draws integer inputs at random otherwise, and a random
        ``cu_seqlens`` describes no packing, so every candidate would time a
        kernel that scans nothing.
        """
        from tilelang.utils.device import get_current_device

        batch, heads = self.batch, self.heads
        dim_qk = self.dim_nope + self.dim_pe
        dim_nope, dim_pe, dim_v = self.dim_nope, self.dim_pe, self.dim_v
        dtype = self.dtype
        tokens_per_request = 256
        total = batch * tokens_per_request

        def supply_prog(params):
            if len(params) != 7:
                raise RuntimeError(
                    "autotuning MLAVarlenPrefillFwdKernel expects q, k_nope, k_pe, v, "
                    f"cu_seqlens and two outputs, got {len(params)} parameters"
                )
            device = get_current_device()
            cu_seqlens = torch.arange(
                0, total + 1, tokens_per_request, dtype=torch.int32, device=device
            )
            return [
                torch.randn(total, heads, dim_qk, dtype=dtype, device=device),
                torch.randn(total, heads, dim_nope, dtype=dtype, device=device),
                torch.randn(total, dim_pe, dtype=dtype, device=device),
                torch.randn(total, heads, dim_v, dtype=dtype, device=device),
                cu_seqlens,
            ]

        return supply_prog

    @property
    def autotune_supply_prog(self) -> Callable:
        return self._supply_prog

    @staticmethod
    def _shared_bytes(
        batch: int, dim_nope: int, dim_pe: int, dim_v: int, itemsize: int, config: dict
    ) -> int:
        """Shared memory *config* allocates, buffer by buffer as TileLang aligns them; the K and
        V loads do not multi-buffer, so ``num_stages`` adds nothing."""
        block_m, block_n = config["block_m"], config["block_n"]
        buffers = [
            block_m * dim_nope * itemsize,
            block_m * dim_pe * itemsize,
            block_n * dim_nope * itemsize,
            block_n * dim_pe * itemsize,
            block_n * dim_v * itemsize,
            4 * (batch + 1),
        ]
        if _stages_score_tile(block_m, config["threads"]):
            buffers.append(block_m * block_n * itemsize)
        align = SHARED_BUFFER_ALIGN_BYTES
        return sum(-(-b // align) * align for b in buffers)

    @property
    def default_config(self) -> dict:
        cap = get_shared_memory_optin(self.device_index)
        return next(
            (
                c
                for c in _CANDIDATES
                if self._shared_bytes(
                    self.batch, self.dim_nope, self.dim_pe, self.dim_v, self.dtype.itemsize, c
                )
                <= cap
            ),
            _NARROW,
        ).copy()

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [64, 128]
        block_n = [64, 128, 256]
        num_stages = [1, 2, 3]
        threads = [128, 256, 384]
        return [
            {"block_m": c[0], "block_n": c[1], "num_stages": c[2], "threads": c[3]}
            for c in itertools.product(block_m, block_n, num_stages, threads)
        ]

    def forward(
        self,
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend each query row to the keys its own request's mask admits.

        Args:
            q: ``(total_tokens, heads, dim_nope + dim_pe)``.
            k_nope: ``(total_tokens, heads, dim_nope)``.
            k_pe: ``(total_tokens, dim_pe)``, one row per token for every head.
            v: ``(total_tokens, heads, dim_v)``.
            cu_seqlens: ``(batch + 1)`` packed request offsets.

        Returns:
            The ``(total_tokens, heads, dim_v)`` output and its float32
            ``(total_tokens, heads)`` log-sum-exp.
        """
        self._require_cuda(q=q, k_nope=k_nope, k_pe=k_pe, v=v)
        program = self.kernel(
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
        )
        return program(q, k_nope, k_pe, v, cu_seqlens)
