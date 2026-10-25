"""MLA prefill backward after the latent is decompressed, fixed-length and packed.

Every head has its own ``dim_qk``-wide query and key and its own ``dim_v``-wide value, so
the contractions run at their own widths: the scores, ``dQ`` and ``dK`` over ``dim_qk``,
``dP`` and ``dV`` over ``dim_v``. Three launches serve one call:

1. ``delta = rowsum(o * do)``, head-major like ``lse``.
2. One CTA per (request, key block, head) keeps its keys and values in shared memory and
   ``dK`` and ``dV`` in registers, walks the query blocks its mask admits, and adds each
   step's ``dQ`` partial to a float32 accumulator.
3. The accumulator is scaled and cast to the input dtype.

A fixed-length call is the packed one with every request ``seqlen`` long; only the layout
of ``lse`` and ``delta`` differs, batch-major ``(B, H, S)`` against head-major ``(H, T)``.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    MLABwdCall,
    MLADenseBwdInterface,
    MLAVarlenBwdInterface,
)
from tileops._kernels.constants import LOG2E
from tileops._kernels.grouped_tiling import GroupTiling
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel

__all__ = ["MLADenseBwdKernel", "MLAVarlenBwdKernel"]


@functools.lru_cache(maxsize=32)
def _delta_kernel(batch: int, heads: int, dim_v: int, dense_seqlen: int, dtype: str) -> Callable:
    """``delta[.., h, row] = sum_d o[row, h, d] * do[row, h, d]`` in float32."""
    block = 32

    @tilelang.jit(
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _delta_func() -> Callable:
        total = T.dynamic("total_q")
        delta_shape = (batch, heads, dense_seqlen) if dense_seqlen else (heads, total)

        @T.prim_func
        def _delta_main(
            o: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
            do: T.Tensor((total, heads, dim_v), dtype),  # type: ignore
            delta: T.Tensor(delta_shape, "float32"),  # type: ignore
        ) -> None:
            with T.Kernel(heads, T.ceildiv(total, block), threads=128) as (h, bx):
                o_f = T.alloc_fragment([block, dim_v], "float32")
                do_f = T.alloc_fragment([block, dim_v], "float32")
                rows = T.alloc_fragment([block], "float32")
                T.copy(o[bx * block : (bx + 1) * block, h, :], o_f)
                T.copy(do[bx * block : (bx + 1) * block, h, :], do_f)
                for i, d in T.Parallel(block, dim_v):
                    o_f[i, d] = o_f[i, d] * do_f[i, d]
                T.reduce_sum(o_f, rows, 1)
                for i in T.Parallel(block):
                    row = bx * block + i
                    if row < total:
                        if dense_seqlen:
                            delta[row // dense_seqlen, h, row % dense_seqlen] = rows[i]
                        else:
                            delta[h, row] = rows[i]

        return _delta_main

    return _delta_func()


@functools.lru_cache(maxsize=32)
def _dq_land_kernel(heads: int, dim_qk: int, scale: float, dtype: str) -> Callable:
    """``dq = dq_acc * scale`` in the input dtype."""
    block = 64

    @tilelang.jit(
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _dq_land_func() -> Callable:
        total = T.dynamic("total_q")

        @T.prim_func
        def _dq_land_main(
            dq_acc: T.Tensor((total, heads, dim_qk), "float32"),  # type: ignore
            dq: T.Tensor((total, heads, dim_qk), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(total, block), heads, threads=128) as (bx, h):
                acc = T.alloc_fragment([block, dim_qk], "float32")
                T.copy(dq_acc[bx * block : (bx + 1) * block, h, :], acc)
                for i, d in T.Parallel(block, dim_qk):
                    acc[i, d] = acc[i, d] * scale
                T.copy(acc, dq[bx * block : (bx + 1) * block, h, :])

        return _dq_land_main

    return _dq_land_func()


@functools.lru_cache(maxsize=32)
def _bwd_kernel(
    batch: int,
    heads: int,
    dim_qk: int,
    dim_v: int,
    dense_seqlen_q: int,
    dense_seqlen_kv: int,
    is_causal: bool,
    scale: float,
    dtype: str,
) -> Callable:
    """The key-stationary backward; ``dense_seqlen_*`` are ``0`` for a packed call."""
    accum = "float32"
    scale_log2 = scale * LOG2E
    dense = dense_seqlen_q > 0

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _bwd_func(block_n: int, block_m: int, num_stages: int, threads: int) -> Callable:
        total_q = T.dynamic("total_q")
        total_kv = T.dynamic("total_kv")
        if dense:
            stat_shape = (batch, heads, dense_seqlen_q)
            num_tiles = batch * T.ceildiv(dense_seqlen_kv, block_n)
        else:
            stat_shape = (heads, total_q)
            tiling = GroupTiling(batch, block_n)
            num_tiles = tiling.tile_upper_bound(total_kv)

        @T.prim_func
        def _bwd_main(
            q: T.Tensor((total_q, heads, dim_qk), dtype),  # type: ignore
            k: T.Tensor((total_kv, heads, dim_qk), dtype),  # type: ignore
            v: T.Tensor((total_kv, heads, dim_v), dtype),  # type: ignore
            do: T.Tensor((total_q, heads, dim_v), dtype),  # type: ignore
            lse: T.Tensor(stat_shape, accum),  # type: ignore
            delta: T.Tensor(stat_shape, accum),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
            dq_acc: T.Tensor((total_q, heads, dim_qk), accum),  # type: ignore
            dk: T.Tensor((total_kv, heads, dim_qk), dtype),  # type: ignore
            dv: T.Tensor((total_kv, heads, dim_v), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(num_tiles, heads, threads=threads) as (tile, h):
                k_s = T.alloc_shared([block_n, dim_qk], dtype)
                v_s = T.alloc_shared([block_n, dim_v], dtype)
                q_s = T.alloc_shared([block_m, dim_qk], dtype)
                do_s = T.alloc_shared([block_m, dim_v], dtype)
                ds_s = T.alloc_shared([block_n, block_m], dtype)
                lse_s = T.alloc_shared([block_m], accum)
                delta_s = T.alloc_shared([block_m], accum)
                tile_cum = T.alloc_shared([batch + 1], "int32")
                st = T.alloc_fragment([block_n, block_m], accum)
                dpt = T.alloc_fragment([block_n, block_m], accum)
                p16 = T.alloc_fragment([block_n, block_m], dtype)
                ds16 = T.alloc_fragment([block_n, block_m], dtype)
                dk_f = T.alloc_fragment([block_n, dim_qk], accum)
                dv_f = T.alloc_fragment([block_n, dim_v], accum)
                dq_f = T.alloc_fragment([block_m, dim_qk], accum)
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                request = T.alloc_local([1], "int32")
                n0 = T.alloc_local([1], "int32")
                live = T.alloc_local([1], "int32")
                if dense:
                    request[0] = tile // T.ceildiv(dense_seqlen_kv, block_n)
                    n0[0] = (tile % T.ceildiv(dense_seqlen_kv, block_n)) * block_n
                    live[0] = 1
                else:
                    tiling.cumsum_offsets(cu_seqlens_kv, tile_cum)
                    live[0] = T.if_then_else(tile < tile_cum[batch], 1, 0)
                    if live[0] == 1:
                        tiling.decode(tile, tile_cum, lo, hi, request, n0)
                if live[0] == 1:
                    b = request[0]
                    q_start = cu_seqlens_q[b]
                    kv_start = cu_seqlens_kv[b]
                    len_q = cu_seqlens_q[b + 1] - q_start
                    len_kv = cu_seqlens_kv[b + 1] - kv_start
                    shift = len_kv - len_q
                    for i, d in T.Parallel(block_n, dim_qk):
                        k_s[i, d] = T.if_then_else(
                            n0[0] + i < len_kv, k[kv_start + n0[0] + i, h, d], T.cast(0, dtype)
                        )
                    for i, d in T.Parallel(block_n, dim_v):
                        v_s[i, d] = T.if_then_else(
                            n0[0] + i < len_kv, v[kv_start + n0[0] + i, h, d], T.cast(0, dtype)
                        )
                    T.clear(dk_f)
                    T.clear(dv_f)
                    loop_st = T.max(0, T.floordiv(n0[0] - shift, block_m)) if is_causal else 0
                    loop_ed = T.ceildiv(len_q, block_m)
                    for kq in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                        m0 = kq * block_m
                        for i, d in T.Parallel(block_m, dim_qk):
                            q_s[i, d] = T.if_then_else(
                                m0 + i < len_q, q[q_start + m0 + i, h, d], T.cast(0, dtype)
                            )
                        for i, d in T.Parallel(block_m, dim_v):
                            do_s[i, d] = T.if_then_else(
                                m0 + i < len_q, do[q_start + m0 + i, h, d], T.cast(0, dtype)
                            )
                        for i in T.Parallel(block_m):
                            if dense:
                                lse_s[i] = T.if_then_else(m0 + i < len_q, lse[b, h, m0 + i], 0)
                                delta_s[i] = T.if_then_else(m0 + i < len_q, delta[b, h, m0 + i], 0)
                            else:
                                lse_s[i] = T.if_then_else(
                                    m0 + i < len_q, lse[h, q_start + m0 + i], 0
                                )
                                delta_s[i] = T.if_then_else(
                                    m0 + i < len_q, delta[h, q_start + m0 + i], 0
                                )
                        T.clear(st)
                        T.gemm(k_s, q_s, st, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.clear(dpt)
                        T.gemm(v_s, do_s, dpt, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(block_n, block_m):
                            if is_causal:
                                st[i, j] = T.if_then_else(
                                    (m0 + j < len_q)
                                    & (n0[0] + i < len_kv)
                                    & (n0[0] + i <= m0 + j + shift),
                                    T.exp2(st[i, j] * scale_log2 - lse_s[j] * LOG2E),
                                    0,
                                )
                            else:
                                st[i, j] = T.if_then_else(
                                    (m0 + j < len_q) & (n0[0] + i < len_kv),
                                    T.exp2(st[i, j] * scale_log2 - lse_s[j] * LOG2E),
                                    0,
                                )
                        T.copy(st, p16)
                        T.gemm(p16, do_s, dv_f, policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(block_n, block_m):
                            ds16[i, j] = st[i, j] * (dpt[i, j] - delta_s[j])
                        T.gemm(ds16, q_s, dk_f, policy=T.GemmWarpPolicy.FullRow)
                        T.copy(ds16, ds_s)
                        T.clear(dq_f)
                        T.gemm(ds_s, k_s, dq_f, transpose_A=True)
                        for i, d in T.Parallel(block_m, dim_qk):
                            if m0 + i < len_q:
                                T.atomic_add(dq_acc[q_start + m0 + i, h, d], dq_f[i, d])
                    for i, d in T.Parallel(block_n, dim_qk):
                        if n0[0] + i < len_kv:
                            dk[kv_start + n0[0] + i, h, d] = T.cast(dk_f[i, d] * scale, dtype)
                    for i, d in T.Parallel(block_n, dim_v):
                        if n0[0] + i < len_kv:
                            dv[kv_start + n0[0] + i, h, d] = T.cast(dv_f[i, d], dtype)

        return _bwd_main

    return _bwd_func


class _MLABwdKernelBase(Kernel):
    """The three launches of one backward call; the subclasses differ in layout only."""

    supported_archs: list[int] = [90]
    general: bool = True

    @classmethod
    def refusal(cls, call: MLABwdCall) -> Optional[str]:
        if call.dtype not in ATTENTION_DTYPES:
            return f"runs float16 and bfloat16, got {call.dtype}"
        if call.dim_qk % 16 != 0 or call.dim_v % 16 != 0:
            return (
                f"contracts every head dimension in 16-wide steps, got {call.dim_qk}, {call.dim_v}"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: MLABwdCall) -> Entry:
        args = dict(
            batch=call.batch,
            heads=call.heads,
            dim_qk=call.dim_qk,
            dim_v=call.dim_v,
            seqlen_q=call.seqlen_q,
            seqlen_kv=call.seqlen_kv,
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
        dim_qk: int,
        dim_v: int,
        seqlen_q: int,
        seqlen_kv: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        config: Optional[dict] = None,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch, self.heads = batch, heads
        self.dim_qk, self.dim_v = dim_qk, dim_v
        self.seqlen_q, self.seqlen_kv = seqlen_q, seqlen_kv
        self.is_causal = is_causal
        self.dtype = dtype
        self.scale = dim_qk**-0.5 if sm_scale is None else sm_scale
        self.kernel = _bwd_kernel(
            batch,
            heads,
            dim_qk,
            dim_v,
            seqlen_q,
            seqlen_kv,
            is_causal,
            self.scale,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"block_n": 128, "block_m": 32, "num_stages": 2, "threads": 256}

    def _run(self, q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv):
        """Run the three launches over packed ``(T, H, D)`` tensors."""
        prep, main, land = self.program(self._compile)
        delta = prep(o, do)
        dq_acc = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        main(q, k, v, do, lse, delta, cu_seqlens_q, cu_seqlens_kv, dq_acc, dk, dv)
        dq = land(dq_acc)
        return dq, dk, dv

    def _compile(self, cfg: dict):
        """The three programs of one backward call: the delta, the main pass compiled from
        *cfg*, and the dQ scaling."""
        prep = _delta_kernel(self.batch, self.heads, self.dim_v, self.seqlen_q, self.dtype_str)
        main = self.kernel(cfg["block_n"], cfg["block_m"], cfg["num_stages"], cfg["threads"])
        land = _dq_land_kernel(self.heads, self.dim_qk, self.scale, self.dtype_str)
        return CompiledProgram(prep), CompiledProgram(main), CompiledProgram(land)


class MLADenseBwdKernel(_MLABwdKernelBase, MLADenseBwdInterface):
    """MLA prefill backward over fixed-length requests (BSHD)."""

    def forward(self, q, k, v, o, do, lse):
        self._require_devices(q=q, k=k, v=v, o=o, do=do, lse=lse)
        batch, seqlen_q, heads, _ = q.shape
        seqlen_kv = k.shape[1]
        cu_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
        cu_kv = torch.arange(
            0, (batch + 1) * seqlen_kv, seqlen_kv, dtype=torch.int32, device=q.device
        )
        dq, dk, dv = self._run(
            q.view(batch * seqlen_q, heads, -1),
            k.view(batch * seqlen_kv, heads, -1),
            v.view(batch * seqlen_kv, heads, -1),
            o.view(batch * seqlen_q, heads, -1),
            do.view(batch * seqlen_q, heads, -1),
            lse,
            cu_q,
            cu_kv,
        )
        return dq.view(q.shape), dk.view(k.shape), dv.view(v.shape)


class MLAVarlenBwdKernel(_MLABwdKernelBase, MLAVarlenBwdInterface):
    """MLA prefill backward over packed requests (THD)."""

    def forward(self, q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv):
        self._require_devices(q=q, k=k, v=v, o=o, do=do, lse=lse)
        return self._run(q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv)
