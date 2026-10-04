import functools
import itertools
import math
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import (
    AttentionCall,
    GQABwdInterface,
    GQAPreprocessBwdInterface,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin

__all__ = ["FlashAttnBwdPreprocessKernel", "GQABwdMmaKernel", "GQABwdWgmmaPipelinedKernel"]

_ROWS_PER_BLOCK = 64


@functools.lru_cache(maxsize=32)
@tilelang.jit
def _flashattn_bwd_preprocess_kernel(
    batch: int, heads: int, seq_len: int, dim: int, dtype: str
) -> Callable:
    accum_dtype = "float"
    shape = (batch, seq_len, heads, dim)
    blk = _ROWS_PER_BLOCK
    # TileLang finds no fragment layout for a row sum over a width that is not a power of
    # two, head dim 96 for one, so the rows are summed over power-of-two column tiles.
    cols = math.gcd(dim, 64)

    @T.prim_func
    def flash_bwd_prep(
        o: T.Tensor(shape, dtype),  # type: ignore
        do: T.Tensor(shape, dtype),  # type: ignore
        delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        # Zeroed in head-major order, so each block clears one contiguous span.
        dq_accum: T.Tensor([batch, heads, seq_len, dim], accum_dtype),  # type: ignore
    ) -> None:
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch, threads=128) as (bx, by, bz):
            rows = slice(by * blk, (by + 1) * blk)
            o_frag = T.alloc_fragment([blk, cols], dtype)
            do_frag = T.alloc_fragment([blk, cols], dtype)
            acc = T.alloc_fragment([blk, cols], accum_dtype)
            delta_frag = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for c in T.serial(dim // cols):
                T.copy(o[bz, rows, bx, c * cols : (c + 1) * cols], o_frag)
                T.copy(do[bz, rows, bx, c * cols : (c + 1) * cols], do_frag)
                for i, j in T.Parallel(blk, cols):
                    acc[i, j] += T.cast(o_frag[i, j], accum_dtype) * T.cast(
                        do_frag[i, j], accum_dtype
                    )
            T.reduce_sum(acc, delta_frag, 1)
            T.copy(delta_frag, delta[bz, bx, rows])
            T.clear(acc)
            for c in T.serial(dim // cols):
                T.copy(acc, dq_accum[bz, bx, rows, c * cols : (c + 1) * cols])

    return flash_bwd_prep


class FlashAttnBwdPreprocessKernel(Kernel, GQAPreprocessBwdInterface):
    """Row-wise ``delta = rowsum(o * do)`` for the GQA/MHA backward pass; also zeroes
    the f32 ``dq`` accumulator the backward kernel adds into.

    The launch geometry is fixed, so ``default_config`` is empty and
    ``autotune_configs`` is undefined: a tuning request leaves it as built.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        seq_len: Sequence length.
        dim: Head dimension.
        dtype: Torch dtype of ``o`` / ``do``.
        config: Optional config dict. This kernel exposes no tunable knobs.
        tune: Whether to autotune. No-op for this kernel; see above.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        args = (call.batch, call.heads, call.max_seqlen_q, call.dim, call.dtype)
        return args, lambda: cls(*args)

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = dtype

        self.kernel = _flashattn_bwd_preprocess_kernel(
            self.batch, self.heads, self.seq_len, self.dim, self.dtype_str
        )
        self.init_config(config, tune)

    def forward(self, o: torch.Tensor, do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(delta, dq_accum)``, the second zero-filled.

        ``dq_accum`` holds as many f32 elements as ``o``; the backward kernel that adds
        into it decides their order.
        """
        delta = torch.empty(
            (self.batch, self.heads, self.seq_len), dtype=torch.float32, device=o.device
        )
        dq_accum = torch.empty(o.numel(), dtype=torch.float32, device=o.device)
        self.kernel(o, do, delta, dq_accum.view(self.batch, self.heads, self.seq_len, self.dim))
        return delta, dq_accum.view(o.shape)


@functools.lru_cache(maxsize=32)
@tilelang.jit
def _flashattn_bwd_postprocess_kernel(
    batch: int, heads: int, seq_len: int, dim: int, dtype: str
) -> Callable:
    shape = (batch, seq_len, heads, dim)
    blk = _ROWS_PER_BLOCK

    @T.prim_func
    def flash_bwd_post(
        dq_accum: T.Tensor(shape, "float"),  # type: ignore
        dq: T.Tensor(shape, dtype),  # type: ignore
    ) -> None:
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch, threads=128) as (bx, by, bz):
            acc = T.alloc_fragment([blk, dim], "float")
            out = T.alloc_fragment([blk, dim], dtype)
            T.copy(dq_accum[bz, by * blk : (by + 1) * blk, bx, :], acc)
            T.copy(acc, out)
            T.copy(out, dq[bz, by * blk : (by + 1) * blk, bx, :])

    return flash_bwd_post


@functools.lru_cache(maxsize=32)
def _gqa_bwd_wgmma_pipelined_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str = "float16",
) -> Callable:
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"
    # One query head per key/value head writes its dK/dV outright; a shared one adds into f32.
    dkv_dtype = dtype if groups == 1 else accum_dtype

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_bwd_wgmma_pipelined_func(
        block_m: int, block_n: int, num_stages: int, threads: int
    ) -> Callable:
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        @T.prim_func
        def _gqa_bwd_wgmma_pipelined_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            do: T.Tensor(q_shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            dq: T.Tensor(q_shape, accum_dtype),  # type: ignore
            dk: T.Tensor(kv_shape, dkv_dtype),  # type: ignore
            dv: T.Tensor(kv_shape, dkv_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(heads, T.ceildiv(seq_len, block_m), batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                k_shared = T.alloc_shared([block_m, dim], dtype)
                dst_shared = T.alloc_shared([block_m, block_n], dtype)
                q_frag = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_m, dim], dtype)
                qkt = T.alloc_fragment([block_m, block_n], accum_dtype)
                dst = T.alloc_fragment([block_m, block_n], accum_dtype)
                qkt_cast = T.alloc_fragment([block_m, block_n], dtype)
                dst_cast = T.alloc_fragment([block_m, block_n], dtype)
                lse_shared = T.alloc_shared([block_n], accum_dtype)
                delta_shared = T.alloc_shared([block_n], accum_dtype)
                do_shared = T.alloc_shared([block_n, dim], dtype)
                dv_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dk_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dq_frag = T.alloc_fragment([block_n, dim], accum_dtype)
                dq_shared = T.alloc_shared([block_n, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_m, dim], dkv_dtype)
                dk_shared = T.alloc_shared([block_m, dim], dkv_dtype)

                T.annotate_layout(
                    {
                        dq_shared: tilelang.layout.make_swizzled_layout(dq_shared),
                        dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                        dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                    }
                )

                T.copy(k[bz, by * block_m : (by + 1) * block_m, bx // groups, :], k_shared)
                T.copy(v[bz, by * block_m : (by + 1) * block_m, bx // groups, :], v_shared)
                T.clear(dv_frag)
                T.clear(dk_frag)
                stage_hold = T.alloc_local([1], dtype)

                loop_st = T.floordiv(by * block_m, block_n) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_n)

                for k_idx in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    # Every accumulator is read only after a wait that covers its WGMMA:
                    # a register read of an in-flight accumulator makes ptxas serialize
                    # every WGMMA in the kernel.
                    T.copy(q[bz, k_idx * block_n : (k_idx + 1) * block_n, bx, :], q_frag)
                    T.wgmma_gemm(
                        k_shared,
                        q_frag,
                        qkt,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                    )
                    T.copy(do[bz, k_idx * block_n : (k_idx + 1) * block_n, bx, :], do_shared)
                    T.wgmma_gemm(
                        v_shared,
                        do_shared,
                        dst,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                    )
                    T.copy(lse[bz, bx, k_idx * block_n : (k_idx + 1) * block_n], lse_shared)
                    T.wait_wgmma(1)
                    for i, j in T.Parallel(block_m, block_n):
                        qkt[i, j] = T.exp2(qkt[i, j] * scale - lse_shared[j])
                    if is_causal and k_idx * block_n < (by + 1) * block_m:
                        for i, j in T.Parallel(block_m, block_n):
                            qkt[i, j] = T.if_then_else(
                                by * block_m + i <= k_idx * block_n + j, qkt[i, j], 0
                            )
                    T.copy(qkt, qkt_cast)
                    T.wgmma_gemm(qkt_cast, do_shared, dv_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(delta[bz, bx, k_idx * block_n : (k_idx + 1) * block_n], delta_shared)
                    T.wait_wgmma(1)
                    for i, j in T.Parallel(block_m, block_n):
                        dst_cast[i, j] = qkt[i, j] * (dst[i, j] - delta_shared[j]) * sm_scale
                    T.wgmma_gemm(dst_cast, q_frag, dk_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(dst_cast, dst_shared)
                    T.wgmma_gemm(dst_shared, k_shared, dq_frag, transpose_A=True, clear_accum=True)
                    T.wait_wgmma(0)
                    T.copy(dq_frag, dq_shared)
                    T.atomic_add(
                        dq[bz, k_idx * block_n : (k_idx + 1) * block_n, bx, :],
                        dq_shared,
                        use_tma=True,
                    )
                    # The pipeline hands a stage back after its buffer's last access, and
                    # a WGMMA issue is not a completed read: touching q_frag and do_shared
                    # after the wait above holds both stages until dK and dV are done.
                    stage_hold[0] = q_frag[0, 0] + do_shared[0, 0]
                rows = slice(by * block_m, (by + 1) * block_m)
                T.copy(dv_frag, dv_shared)
                if groups == 1:
                    T.copy(dv_shared, dv[bz, rows, bx, :])
                else:
                    T.atomic_add(dv[bz, rows, bx // groups, :], dv_shared)
                T.copy(dk_frag, dk_shared)
                if groups == 1:
                    T.copy(dk_shared, dk[bz, rows, bx, :])
                else:
                    T.atomic_add(dk[bz, rows, bx // groups, :], dk_shared)

        return _gqa_bwd_wgmma_pipelined_main

    return _gqa_bwd_wgmma_pipelined_func


@functools.lru_cache(maxsize=32)
def _gqa_bwd_mma_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str = "float16",
) -> Callable:
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"
    # One query head per key/value head writes its dK/dV outright; a shared one adds into f32.
    dkv_dtype = dtype if groups == 1 else accum_dtype

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_bwd_mma_func(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        @T.prim_func
        def _gqa_bwd_mma_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            do: T.Tensor(q_shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            dq: T.Tensor(q_shape, accum_dtype),  # type: ignore
            dk: T.Tensor(kv_shape, dkv_dtype),  # type: ignore
            dv: T.Tensor(kv_shape, dkv_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(heads, T.ceildiv(seq_len, block_m), batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                k_shared = T.alloc_shared([block_m, dim], dtype)
                dst_shared = T.alloc_shared([block_m, block_n], dtype)
                q_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_m, dim], dtype)
                qkt = T.alloc_fragment([block_m, block_n], accum_dtype)
                dst = T.alloc_fragment([block_m, block_n], accum_dtype)
                qkt_cast = T.alloc_fragment([block_m, block_n], dtype)
                dst_cast = T.alloc_fragment([block_m, block_n], dtype)
                lse_shared = T.alloc_shared([block_n], accum_dtype)
                delta_shared = T.alloc_shared([block_n], accum_dtype)
                do_shared = T.alloc_shared([block_n, dim], dtype)
                dv_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dk_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dq_frag = T.alloc_fragment([block_n, dim], accum_dtype)
                if groups > 1:
                    dkv_shared = T.alloc_shared([block_m, dim], accum_dtype)
                    T.annotate_layout(
                        {dkv_shared: tilelang.layout.make_swizzled_layout(dkv_shared)}
                    )

                T.copy(k[bz, by * block_m : (by + 1) * block_m, bx // groups, :], k_shared)
                T.copy(v[bz, by * block_m : (by + 1) * block_m, bx // groups, :], v_shared)
                T.clear(dv_frag)
                T.clear(dk_frag)

                loop_st = T.floordiv(by * block_m, block_n) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_n)

                for k_idx in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(q[bz, k_idx * block_n : (k_idx + 1) * block_n, bx, :], q_shared)
                    T.clear(qkt)
                    T.gemm(
                        k_shared, q_shared, qkt, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    T.copy(lse[bz, bx, k_idx * block_n : (k_idx + 1) * block_n], lse_shared)
                    for i, j in T.Parallel(block_m, block_n):
                        qkt[i, j] = T.exp2(qkt[i, j] * scale - lse_shared[j])
                    if is_causal and k_idx * block_n < (by + 1) * block_m:
                        for i, j in T.Parallel(block_m, block_n):
                            qkt[i, j] = T.if_then_else(
                                by * block_m + i <= k_idx * block_n + j, qkt[i, j], 0
                            )
                    T.copy(do[bz, k_idx * block_n : (k_idx + 1) * block_n, bx, :], do_shared)
                    T.clear(dst)
                    T.gemm(
                        v_shared, do_shared, dst, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    T.copy(qkt, qkt_cast)
                    T.gemm(qkt_cast, do_shared, dv_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(delta[bz, bx, k_idx * block_n : (k_idx + 1) * block_n], delta_shared)
                    for i, j in T.Parallel(block_m, block_n):
                        dst_cast[i, j] = qkt[i, j] * (dst[i, j] - delta_shared[j]) * sm_scale
                    T.gemm(dst_cast, q_shared, dk_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(dst_cast, dst_shared)
                    T.clear(dq_frag)
                    T.gemm(dst_shared, k_shared, dq_frag, transpose_A=True)
                    for i, j in T.Parallel(block_n, dim):
                        if k_idx * block_n + i < seq_len:
                            T.atomic_add(dq[bz, k_idx * block_n + i, bx, j], dq_frag[i, j])
                rows = slice(by * block_m, (by + 1) * block_m)
                if groups == 1:
                    # The loop is done with k and v, so their tiles stage dV and dK.
                    T.copy(dv_frag, v_shared)
                    T.copy(v_shared, dv[bz, rows, bx, :])
                    T.copy(dk_frag, k_shared)
                    T.copy(k_shared, dk[bz, rows, bx, :])
                else:
                    T.copy(dv_frag, dkv_shared)
                    T.atomic_add(dv[bz, rows, bx // groups, :], dkv_shared)
                    T.copy(dk_frag, dkv_shared)
                    T.atomic_add(dk[bz, rows, bx // groups, :], dkv_shared)

        return _gqa_bwd_mma_main

    return _gqa_bwd_mma_func


class GQABwdWgmmaPipelinedKernel(Kernel, GQABwdInterface):
    """GQA/MHA backward, one CTA per key block; dQ is added into an f32 buffer in the
    layout of ``q`` and landed in the input dtype by a second launch."""

    supported_archs: list[int] = [90]
    # The implementation behind the specialised ones for this key.
    general: bool = True
    _build = staticmethod(_gqa_bwd_wgmma_pipelined_kernel)

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        args = (
            call.batch,
            call.heads,
            call.heads_kv,
            call.max_seqlen_q,
            call.dim,
            call.is_causal,
            call.dtype,
        )
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = self._build(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.is_causal,
            self.dtype_str,
        )
        self.post_kernel = _flashattn_bwd_postprocess_kernel(
            self.batch, self.heads, self.seq_len, self.dim, self.dtype_str
        )

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 128, "block_n": 64, "num_stages": 2, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [64, 128]
        block_n = [64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, block_n, num_stages, threads))

        return [
            {"block_m": c[0], "block_n": c[1], "num_stages": c[2], "threads": c[3]}
            for c in _configs
        ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
        delta: torch.Tensor,
        dq_accum: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(dq, dk, dv)`` in the input dtype; ``dq_accum`` arrives zero-filled."""
        dq_accum = dq_accum.view(q.shape)
        grouped = self.heads_kv != self.heads
        make = (
            functools.partial(torch.zeros_like, dtype=torch.float32)
            if grouped
            else torch.empty_like
        )
        dk, dv = make(k), make(v)
        self.kernel(**self.config)(q, k, v, do, lse, delta, dq_accum, dk, dv)
        if grouped:
            dk, dv = dk.to(q.dtype), dv.to(q.dtype)
        dq = torch.empty_like(q)
        self.post_kernel(dq_accum, dq)
        return dq, dk, dv


class GQABwdMmaKernel(GQABwdWgmmaPipelinedKernel):
    """The same backward on MMA, for GPUs without WGMMA; dQ is added element by element."""

    supported_archs: list[int] = [80, 86, 89]
    general: bool = False
    _build = staticmethod(_gqa_bwd_mma_kernel)
    # Key rows of the default block: the row-split GEMMs give each of its four warps 16.
    _BLOCK_M = 64

    @classmethod
    def applies(cls, call: AttentionCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Require complete MMA contractions and enough shared memory for the narrowest tile."""
        if call.dim % 16 != 0:
            return f"head dim must be a multiple of 16 for the MMA contraction, got {call.dim}"
        if not call.smem_budget:
            return None
        need = cls._live_bytes(call.dim, cls._query_blocks(call.dim)[-1], call.dtype.itemsize)
        if need <= call.smem_budget:
            return None
        return (
            f"needs at least {need} bytes of shared memory per block at head dim {call.dim} "
            f"in {call.dtype}; the device gives {call.smem_budget}"
        )

    @property
    def default_config(self) -> dict:
        budget = get_shared_memory_optin(self.device_index)
        blocks = self._query_blocks(self.dim)
        grouped = self.heads != self.heads_kv
        block_n = next(
            (
                n
                for n in blocks
                if self._shared_bytes(self.dim, n, self.dtype.itemsize, grouped) <= budget
            ),
            blocks[-1],
        )
        return {"block_m": self._BLOCK_M, "block_n": block_n, "num_stages": 1, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        # The row-split GEMMs give each warp 16 rows, so eight warps need a 128-row block.
        return [
            {"block_m": m, "block_n": n, "num_stages": s, "threads": t}
            for m, n, s in itertools.product([64, 128], [16, 32, 64], [1, 2])
            for t in ((128, 256) if m == 128 else (128,))
        ]

    @staticmethod
    def _query_blocks(dim: int) -> tuple[int, ...]:
        """Query blocks the default chooses from, widest first. At 16 rows the dQ product's
        four warps split dim instead, 8 columns at a time."""
        widest = 64 if dim <= 64 else 32
        return (widest, 16) if dim % 32 == 0 else (widest,)

    @classmethod
    def _shared_bytes(cls, dim: int, block_n: int, itemsize: int, grouped: bool) -> int:
        """Upper bound on the default program's shared memory: every buffer it allocates, the
        key, value, query and dO rows, dS, the fp32 lse and delta, and with grouped KV heads
        the fp32 dV/dK staging tile."""
        m = cls._BLOCK_M
        loop = (2 * m * dim + 2 * block_n * dim + m * block_n) * itemsize + 2 * block_n * 4
        return loop + (m * dim * 4 if grouped else 0)

    @classmethod
    def _live_bytes(cls, dim: int, block_n: int, itemsize: int) -> int:
        """Lower bound on the default program's shared memory: the key, value, query and dO
        rows live together at the dV product."""
        return 2 * (cls._BLOCK_M + block_n) * dim * itemsize
