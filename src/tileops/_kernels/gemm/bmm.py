"""Batched GEMM kernel for BMMFwdOp.

Shapes are strict 3D-3D — ``a``: $[B \\times M \\times K]$, ``b``: $[B \\times K \\times N]$, ``c``: $[B \\times M \\times N]$.
"""

from collections.abc import Callable
from typing import NamedTuple

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops._kernels.gemm.call_spec import (
    BMMCall,
    BMMFP8Call,
    BMMFP8FwdInterface,
    BMMFP8TransposeCall,
    BMMFP8TransposeFwdInterface,
    BMMFwdInterface,
)
from tileops._kernels.gemm.persistent.heuristics import GEMMType
from tileops._kernels.gemm.persistent.template import GEMMTemplate
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = [
    "BMMFP8Kernel",
    "BMMFP8PersistentKernel",
    "BMMFP8TransposeKernel",
    "BMMFP8WSKernel",
    "BMMKernel",
    "BMMPersistentKernel",
    "BMMPingPongKernel",
]


@target_cache(maxsize=64)
def _bmm_kernel(batch: int, m: int, n: int, k: int, dtype: str = "float16") -> Callable:
    """Pipelined batched GEMM on a ``(ceildiv(n, block_n), ceildiv(m, block_m), batch)`` grid,
    accumulating in fp32; ``m`` and ``n`` need not be multiples of the block sizes.

    Args:
        batch: Number of independent GEMM problems (grid.z).
        m: Rows of each ``A[b]`` / ``C[b]``.
        n: Columns of each ``B[b]`` / ``C[b]``.
        k: Contraction dim shared across all batches.
        dtype: Activation / weight dtype string (``"float16"`` or
            ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n, block_k,
        num_stages, threads)``, and ``a_strides`` and ``c_strides``, the element strides of
        ``A`` and ``C`` ending in 1 or empty for contiguous storage, returns the compiled
        ``prim_func``, which takes ``(a, b, c)``.
    """
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
    )
    def _bmm_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        threads: int = 128,
        a_strides: tuple = (),
        c_strides: tuple = (),
    ) -> Callable:
        a_param = (
            T.StridedTensor((batch, m, k), a_strides, dtype)
            if a_strides
            else T.Tensor((batch, m, k), dtype)
        )
        c_param = (
            T.StridedTensor((batch, m, n), c_strides, dtype)
            if c_strides
            else T.Tensor((batch, m, n), dtype)
        )
        if (
            (block_m, block_n, block_k, threads) == (16, 64, 64, 64)
            and k == 128
            and m % 16 == 0
            and n % 64 == 0
        ):

            @T.prim_func
            def two_panels(
                a: a_param,
                b: T.Tensor((batch, k, n), dtype),
                c: c_param,
            ):
                with T.Kernel(n // block_n, m // block_m, batch, threads=threads) as (bx, by, bz):
                    a0 = T.alloc_shared((block_m, block_k), dtype)
                    b0 = T.alloc_shared((block_k, block_n), dtype)
                    a1 = T.alloc_shared((block_m, block_k), dtype)
                    b1 = T.alloc_shared((block_k, block_n), dtype)
                    acc = T.alloc_fragment((block_m, block_n), "float32")
                    T.async_copy(a[bz, by * block_m, 0], a0)
                    T.async_copy(b[bz, 0, bx * block_n], b0)
                    T.async_copy(a[bz, by * block_m, block_k], a1)
                    T.async_copy(b[bz, block_k, bx * block_n], b1)
                    T.clear(acc)
                    # Issue both panels before computing the first one.
                    T.ptx_wait_group(2)
                    T.sync_threads()
                    T.gemm(a0, b0, acc)
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    T.gemm(a1, b1, acc)
                    T.copy(acc, c[bz, by * block_m, bx * block_n])

            return two_panels

        @T.prim_func
        def _bmm_main(
            a: a_param,
            b: T.Tensor((batch, k, n), dtype),
            c: c_param,
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                a_smem = T.alloc_shared((block_m, block_k), dtype)
                b_smem = T.alloc_shared((block_k, block_n), dtype)
                c_smem = T.alloc_shared((block_m, block_n), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                        b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                        c_smem: tilelang.layout.make_swizzled_layout(c_smem),
                    }
                )

                # L2 rasterization: walk (bx, by) in panels of 10 blocks along N so
                # neighbouring waves reuse the same A/B rows in L2.
                T.use_swizzle(10, enable=True)

                T.clear(c_local)
                m_start = by * block_m
                n_start = bx * block_n

                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = ki * block_k
                    T.copy(
                        a[bz, m_start : m_start + block_m, k_start : k_start + block_k],
                        a_smem,
                    )
                    T.copy(
                        b[bz, k_start : k_start + block_k, n_start : n_start + block_n],
                        b_smem,
                    )
                    T.gemm(a_smem, b_smem, c_local, policy=T.GemmWarpPolicy.FullRow)

                # Epilogue: stage fp32 accum through SMEM before GMEM store.
                # The M/N tail guard breaks coalescing off the wgmma fragment
                # layout; SMEM->GMEM stores stay coalesced under predication.
                T.copy(c_local, c_smem)
                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        c[bz, m_start + i, n_start + j] = c_smem[i, j]

        return _bmm_main

    return _bmm_func


@target_cache(maxsize=32)
def _bmm_fp8_kernel(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
) -> Callable:
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            "tl.disable_warp_specialized": False,
        },
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
    )
    def _bmm_fp8_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        threads: int = 256,
    ) -> Callable:
        k_exact = k % block_k == 0
        n_exact = n % block_n == 0
        m_exact = m % block_m == 0

        @T.prim_func
        def _bmm_fp8_main(
            a: T.Tensor((batch, m, k), dtype),  # type: ignore
            b: T.Tensor((batch, n, k), dtype),  # type: ignore  # N-major B
            scale_a: T.Tensor((1,), "float32"),  # type: ignore
            scale_b: T.Tensor((1,), "float32"),  # type: ignore
            c: T.Tensor((batch, m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_shared = T.alloc_shared((block_m, block_n), out_dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                        c_shared: tilelang.layout.make_swizzled_layout(c_shared),
                    }
                )

                m_start = by * block_m
                n_start = bx * block_n
                T.clear(c_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k
                    if k_exact and m_exact:
                        T.copy(
                            a[bz, m_start : m_start + block_m, k_start : k_start + block_k],
                            a_shared,
                        )
                    else:
                        for i, j in T.Parallel(block_m, block_k):
                            a_shared[i, j] = T.if_then_else(
                                (m_start + i < m) & (k_start + j < k),
                                a[bz, m_start + i, k_start + j],
                                T.cast(0, dtype),
                            )
                    if k_exact and n_exact:
                        T.copy(
                            b[bz, n_start : n_start + block_n, k_start : k_start + block_k],
                            b_shared,
                        )
                    else:
                        for i, j in T.Parallel(block_n, block_k):
                            b_shared[i, j] = T.if_then_else(
                                (n_start + i < n) & (k_start + j < k),
                                b[bz, n_start + i, k_start + j],
                                T.cast(0, dtype),
                            )
                    T.gemm(
                        a_shared,
                        b_shared,
                        c_local,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                for i, j in T.Parallel(block_m, block_n):
                    c_local[i, j] = c_local[i, j] * scale_a[0] * scale_b[0]
                T.copy(c_local, c_shared)
                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        c[bz, m_start + i, n_start + j] = c_shared[i, j]

        return _bmm_fp8_main

    return _bmm_fp8_func


@target_cache(maxsize=32)
def _bmm_fp8_persistent_kernel(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    sm_count: int,
) -> Callable:
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            "tl.disable_warp_specialized": True,
        },
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
    )
    def _bmm_fp8_persistent_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        threads: int = 256,
    ) -> Callable:
        @T.prim_func
        def _bmm_fp8_persistent_main(
            a: T.Tensor((batch, m, k), dtype),  # type: ignore
            b: T.Tensor((batch, n, k), dtype),  # type: ignore  # N-major B
            scale_a: T.Tensor((1,), "float32"),  # type: ignore
            scale_b: T.Tensor((1,), "float32"),  # type: ignore
            c: T.Tensor((batch, m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(sm_count, 1, 1, threads=threads) as (bx, _by, _bz):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_shared = T.alloc_shared((block_m, block_n), out_dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                        c_shared: tilelang.layout.make_swizzled_layout(c_shared),
                    }
                )

                for tile_b, tile_m, tile_n in T.Persistent(
                    [batch, T.ceildiv(m, block_m), T.ceildiv(n, block_n)],
                    wave_size=sm_count,
                    index=bx,
                    group_size=8,
                ):
                    m_start = tile_m * block_m
                    n_start = tile_n * block_n
                    T.clear(c_local)

                    for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                        k_start = kk * block_k
                        T.copy(
                            a[tile_b, m_start : m_start + block_m, k_start : k_start + block_k],
                            a_shared,
                        )
                        T.copy(
                            b[tile_b, n_start : n_start + block_n, k_start : k_start + block_k],
                            b_shared,
                        )
                        T.gemm(
                            a_shared,
                            b_shared,
                            c_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    for i, j in T.Parallel(block_m, block_n):
                        c_local[i, j] = c_local[i, j] * scale_a[0] * scale_b[0]
                    T.copy(c_local, c_shared)
                    T.copy(
                        c_shared,
                        c[tile_b, m_start : m_start + block_m, n_start : n_start + block_n],
                    )

        return _bmm_fp8_persistent_main

    return _bmm_fp8_persistent_func


@target_cache(maxsize=32)
def _bmm_fp8_persistent_ws_kernel(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    sm_count: int,
) -> Callable:
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            "tl.disable_warp_specialized": True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _bmm_fp8_persistent_ws_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 128,
        num_stages: int = 3,
        threads: int = 384,
        group_size_m: int = 8,
    ) -> Callable:
        if threads != 384:
            raise ValueError(
                f"one producer and two consumer warpgroups need threads=384, got {threads}"
            )
        if block_m % 2 or block_m < 128:
            raise ValueError(f"block_m must be even and at least 128, got {block_m}")
        if m % block_m or n % block_n or k % block_k:
            raise ValueError(
                f"the tile must divide the shape: m={m}, n={n}, k={k}, "
                f"block=({block_m}, {block_n}, {block_k})"
            )
        half_m = block_m // 2
        num_pid_m = m // block_m
        num_pid_n = n // block_n
        k_iters = k // block_k
        if k_iters < 2:
            raise ValueError(
                f"the one-deep WGMMA pipeline needs two K tiles, got k={k}, block_k={block_k}"
            )
        total_tiles = batch * num_pid_m * num_pid_n
        grid_cta = min(sm_count, total_tiles)
        max_waves = (total_tiles + grid_cta - 1) // grid_cta + 1

        @T.macro
        def _decode_tile(flat_id, bs, ms, ns_, swz_m, swz_n):
            # flat_id = tile_b * (Mt*Nt) + mn_id; mn_id walks group_size_m-row swizzle groups.
            bs[0] = flat_id // T.int32(num_pid_m * num_pid_n)
            mn_id = flat_id % T.int32(num_pid_m * num_pid_n)
            _gin = T.int32(group_size_m * num_pid_n)
            _fpm = (mn_id // _gin) * T.int32(group_size_m)
            if T.int32(num_pid_m) - _fpm >= T.int32(group_size_m):
                swz_m[0] = _fpm + (mn_id % _gin) % T.int32(group_size_m)
                swz_n[0] = (mn_id % _gin) // T.int32(group_size_m)
            else:
                _gsm = T.int32(num_pid_m) - _fpm
                swz_m[0] = _fpm + (mn_id % _gin) % _gsm
                swz_n[0] = (mn_id % _gin) // _gsm
            ms[0] = swz_m[0] * T.int32(block_m)
            ns_[0] = swz_n[0] * T.int32(block_n)

        @T.macro
        def consumer(wg, a_smem, c_local, c_cast, c_shared, prev_slot, gi_cons: T.Ref, ring, io):
            """Consumer warpgroup ``wg``: WGMMA over rows ``wg * half_m`` up of each tile, then
            its scaled, cast store through its own staging tile and named barrier."""
            B_smem, ab_full, ab_empty, bs, ms, ns_, swz_m, swz_n, pid = ring
            scale_a, scale_b, c = io
            T.inc_max_nreg(240)
            for w in T.serial(max_waves):
                flat_id = T.int32(grid_cta) * w + pid
                if flat_id < T.int32(total_tiles):
                    _decode_tile(flat_id, bs, ms, ns_, swz_m, swz_n)
                    for kk in T.Pipelined(k_iters, num_stages=0):
                        slot = gi_cons % num_stages
                        T.barrier_wait(ab_full[slot], (gi_cons // num_stages) & 1)
                        T.wgmma_gemm(
                            a_smem[slot, :, :],
                            B_smem[slot, :, :],
                            c_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=(kk == 0),
                        )
                        if kk > 0:
                            T.wait_wgmma(1)
                            T.barrier_arrive(ab_empty[prev_slot[0]])
                        prev_slot[0] = slot
                        gi_cons = gi_cons + 1
                    T.wait_wgmma(0)
                    T.barrier_arrive(ab_empty[prev_slot[0]])
                    T.warpgroup_fence_operand(c_local, num_regs=64)
                    for i, j in T.Parallel(half_m, block_n):
                        c_local[i, j] = c_local[i, j] * scale_a[0] * scale_b[0]
                    T.copy(c_local, c_cast)
                    T.sync_threads(barrier_id=4 + wg, arrive_count=128)
                    T.copy(c_cast, c_shared)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=4 + wg, arrive_count=128)
                    T.copy(c_shared, c[bs[0], ms[0] + half_m if wg else ms[0], ns_[0]])

        @T.prim_func
        def _bmm_fp8_persistent_ws_main(
            a: T.Tensor((batch, m, k), dtype),  # type: ignore
            b: T.Tensor((batch, n, k), dtype),  # type: ignore  # N-major B
            scale_a: T.Tensor((1,), "float32"),  # type: ignore
            scale_b: T.Tensor((1,), "float32"),  # type: ignore
            c: T.Tensor((batch, m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(grid_cta, threads=threads) as (pid,):
                A_smem_top = T.alloc_shared((num_stages, half_m, block_k), dtype)
                A_smem_bot = T.alloc_shared((num_stages, half_m, block_k), dtype)
                B_smem = T.alloc_shared((num_stages, block_n, block_k), dtype)

                # Per-WG half-tile fp32 accumulator and dtype cast staging.
                C_local_wg0 = T.alloc_fragment((half_m, block_n), accum_dtype)
                C_local_wg1 = T.alloc_fragment((half_m, block_n), accum_dtype)
                C_local_cast_wg0 = T.alloc_fragment((half_m, block_n), out_dtype)
                C_local_cast_wg1 = T.alloc_fragment((half_m, block_n), out_dtype)
                C_shared_wg0 = T.alloc_shared((half_m, block_n), out_dtype)
                C_shared_wg1 = T.alloc_shared((half_m, block_n), out_dtype)

                bs = T.alloc_local((1,), "int32")  # batch id
                ms = T.alloc_local((1,), "int32")  # M start (rows)
                ns_ = T.alloc_local((1,), "int32")  # N start (cols)

                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                swz_m = T.alloc_local((1,), "int32")
                swz_n = T.alloc_local((1,), "int32")

                T.annotate_layout(
                    {
                        A_smem_top: tilelang.layout.make_swizzled_layout(A_smem_top),
                        A_smem_bot: tilelang.layout.make_swizzled_layout(A_smem_bot),
                        B_smem: tilelang.layout.make_swizzled_layout(B_smem),
                        C_shared_wg0: tilelang.layout.make_swizzled_layout(C_shared_wg0),
                        C_shared_wg1: tilelang.layout.make_swizzled_layout(C_shared_wg1),
                    }
                )

                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)

                tx = T.get_thread_binding()

                # Producer WG: tx < 128
                if tx < 128:
                    T.dec_max_nreg(24)

                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid_cta) * w + pid

                        if flat_id < T.int32(total_tiles):
                            _decode_tile(flat_id, bs, ms, ns_, swz_m, swz_n)

                            for kk in T.Pipelined(k_iters, num_stages=0):
                                k_start = kk * block_k
                                slot = gi_prod % num_stages
                                T.barrier_wait(ab_empty[slot], ((gi_prod // num_stages) & 1) ^ 1)
                                # Top half of A (rows ms..ms+half_m).
                                T.tma_copy(
                                    a[bs[0], ms[0] : ms[0] + half_m, k_start : k_start + block_k],
                                    A_smem_top[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                # Bottom half of A (rows ms+half_m..ms+block_m).
                                T.tma_copy(
                                    a[
                                        bs[0],
                                        ms[0] + half_m : ms[0] + block_m,
                                        k_start : k_start + block_k,
                                    ],
                                    A_smem_bot[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                # Full B tile shared between the two math WGs.
                                T.tma_copy(
                                    b[
                                        bs[0],
                                        ns_[0] : ns_[0] + block_n,
                                        k_start : k_start + block_k,
                                    ],
                                    B_smem[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.barrier_arrive(ab_full[slot])
                                gi_prod = gi_prod + 1

                else:
                    ring = (B_smem, ab_full, ab_empty, bs, ms, ns_, swz_m, swz_n, pid)
                    io = (scale_a, scale_b, c)
                    if tx < 256:
                        consumer(
                            0, A_smem_top, C_local_wg0, C_local_cast_wg0, C_shared_wg0, ps0,
                            gi_cons_0, ring, io,
                        )  # fmt: skip
                    else:
                        consumer(
                            1, A_smem_bot, C_local_wg1, C_local_cast_wg1, C_shared_wg1, ps1,
                            gi_cons_1, ring, io,
                        )  # fmt: skip

        return _bmm_fp8_persistent_ws_main

    return _bmm_fp8_persistent_ws_func


@target_cache(maxsize=32)
def _bmm_fp8_transpose_kernel(batch: int, rows: int, cols: int, dtype: str) -> Callable:
    """Swap the last two axes of a contiguous ``[batch, rows, cols]`` tensor.

    Args:
        batch: Leading axis, untouched.
        rows: Extent of the source's second axis.
        cols: Extent of the source's third axis.
        dtype: TileLang dtype string of both tensors.

    Returns:
        A ``(block, threads)`` builder returning the compiled ``prim_func``.
    """

    @tilelang.jit(out_idx=[-1], compile_flags=[*CUDA_COMPILE_FLAGS])
    def _bmm_fp8_transpose_func(block: int = 64, threads: int = 128) -> Callable:
        exact = rows % block == 0 and cols % block == 0

        @T.prim_func
        def _bmm_fp8_transpose_main(
            src: T.Tensor((batch, rows, cols), dtype),  # type: ignore
            dst: T.Tensor((batch, cols, rows), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(cols, block), T.ceildiv(rows, block), batch, threads=threads
            ) as (bx, by, bz):
                # The store below reads the tile down a column, which conflicts on
                # shared-memory banks unless the layout is swizzled.
                tile = T.alloc_shared((block, block), dtype)
                T.annotate_layout({tile: tilelang.layout.make_swizzled_layout(tile)})
                r0, c0 = by * block, bx * block
                if exact:
                    T.copy(src[bz, r0 : r0 + block, c0 : c0 + block], tile)
                else:
                    for i, j in T.Parallel(block, block):
                        tile[i, j] = T.if_then_else(
                            (r0 + i < rows) & (c0 + j < cols),
                            src[bz, r0 + i, c0 + j],
                            T.cast(0, dtype),
                        )
                # ``i`` innermost keeps the store coalesced: ``dst`` is rows-innermost.
                for j, i in T.Parallel(block, block):
                    if (c0 + j < cols) and (r0 + i < rows):
                        dst[bz, c0 + j, r0 + i] = tile[i, j]

        return _bmm_fp8_transpose_main

    return _bmm_fp8_transpose_func


def _declared_strides(t: torch.Tensor) -> tuple:
    """The strides a program declares for *t*: none where it is contiguous, else its own."""
    return () if t.is_contiguous() else t.stride()


class BMMKernel(Kernel, BMMFwdInterface):
    """Batched dense GEMM kernel.

    Computes ``C[b] = A[b] @ B[b]`` for ``b in [0, batch)`` where
    ``A: [batch, m, k]``, ``B: [batch, k, n]``, ``C: [batch, m, n]``.
    fp16 / bf16 inputs, fp32 accumulation. Grid maps the batch axis to
    ``blockIdx.z`` so all batches run in a single kernel launch.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general = True

    @classmethod
    def refusal(cls, call: BMMCall) -> str | None:
        return cls._k_refusal(call.k)

    @staticmethod
    def _k_refusal(k: int) -> str | None:
        """The tile loop steps K by 16."""
        return None if k % 16 == 0 else f"requires k a multiple of 16, got k={k}"

    @classmethod
    def entry_for(cls, call: BMMCall) -> Entry:
        """Build the shape-specialized classic BMM fallback."""
        index = call.device_index
        identity = (call.batch, call.m, call.n, call.k, call.dtype, index)
        return identity, lambda: cls(
            call.batch,
            call.m,
            call.n,
            call.k,
            call.dtype,
            device_index=index,
        )

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: dict | None = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        reason = self._k_refusal(k)
        if reason is not None:
            raise ValueError(f"BMMKernel {reason}")
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.kernel = _bmm_kernel(batch, m, n, k, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        if (
            self.build_target.arch == 90
            and self.dtype in (torch.float16, torch.bfloat16)
            and (self.m, self.n, self.k) == (128, 128, 128)
            and self.batch <= 8
        ):
            # Builds the two-panel program: both 64-wide K panels load before the first MMA.
            return {"block_m": 16, "block_n": 64, "block_k": 64, "num_stages": 2, "threads": 64}
        # 64x64x64, stages=2, one warpgroup: the tile the manifest shapes share.
        block_k = 64 if self.k % 64 == 0 else (32 if self.k % 32 == 0 else 16)
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": block_k,
            "num_stages": 2,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_k_options = [32, 64]
        if self.k % 32 != 0 and self.k % 16 == 0:
            block_k_options = [16]
        configs = [
            {"block_m": bm, "block_n": bn, "block_k": bk, "num_stages": ns, "threads": 128}
            for bm in [64, 128]
            for bn in [64, 128]
            for bk in block_k_options
            for ns in [2, 3, 4]
        ]
        default = self.default_config
        if default["threads"] == 64:
            # The two-panel program, which the grid above does not reach.
            configs.append(default)
        return [c for c in configs if self.k % c["block_k"] == 0]

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty(self.batch, self.m, self.n, dtype=self.dtype, device=a.device)
        self.program(self._compile, _declared_strides(a), _declared_strides(out))(a, b, out)
        return out

    def _compile(self, cfg: dict, a_strides: tuple, c_strides: tuple) -> Callable:
        """The program ``forward`` launches, compiled from *cfg* for ``a`` and ``out``
        storage of *a_strides* and *c_strides*."""
        return self.kernel(**cfg, a_strides=a_strides, c_strides=c_strides)


class _PersistentBand(NamedTuple):
    """Where the persistent path beats :class:`BMMKernel` on one calibrated board."""

    # The tile ``get_best_config`` picks for this path there. Selection and grid
    # sizing count the same tiles, or the region claimed is not the grid launched.
    tile_m: int
    tile_n: int
    # The path claims a call whose tiles fill 1 / min_wave_denom of a persistent wave.
    min_wave_denom: int


class BMMPersistentKernel(Kernel, BMMFwdInterface):
    """Persistent BMM adapter over :class:`GEMMTemplate`, on a calibrated board.

    The template reads the zero-copy ``[batch, n, k]`` view of public
    ``b[batch, k, n]`` storage. :class:`BMMKernel` serves calls
    :meth:`refusal` refuses; this path takes its configuration from the template selector.
    """

    supported_archs: list[int] = [90]
    # By calibration key; a board without an entry keeps BMMKernel.
    _PERSISTENT_BANDS = {"h200": _PersistentBand(tile_m=128, tile_n=256, min_wave_denom=2)}
    # None takes the template selector's configuration.
    _TEMPLATE_CONFIG: dict | None = None

    @staticmethod
    def _tiles(band: _PersistentBand, batch: int, m: int, n: int) -> int:
        """Output tiles this call launches at the band's tile."""
        return batch * -(-m // band.tile_m) * -(-n // band.tile_n)

    @classmethod
    def refusal(cls, call: BMMCall) -> str | None:
        band = cls._PERSISTENT_BANDS.get(call.calibration)
        if band is None:
            return "has no persistent band for this board"
        # TMA addresses the K-contiguous operands and the output row in 16-byte steps.
        step = 16 // call.dtype.itemsize
        if call.n % step != 0 or call.k % step != 0:
            return f"requires n and k multiples of {step}, got n={call.n}, k={call.k}"
        if cls._tiles(band, call.batch, call.m, call.n) * band.min_wave_denom <= call.sm_count:
            return "does not fill a persistent wave"
        return None

    @staticmethod
    def _full_wave_grid(tiles: int, physical_sms: int) -> int:
        """The largest power-of-two grid when it divides *tiles*, else every SM."""
        power_of_two_grid = 1 << (physical_sms.bit_length() - 1)
        return power_of_two_grid if tiles % power_of_two_grid == 0 else physical_sms

    @classmethod
    def entry_for(cls, call: BMMCall) -> Entry:
        """Build the persistent template specialization for this BMM call."""
        index = call.device_index
        identity = (call.batch, call.m, call.n, call.k, call.dtype, index)
        return identity, lambda: cls(
            call.batch,
            call.m,
            call.n,
            device_index=index,
        )

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        device_index: int | None = None,
        template_config: dict | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch, self.m, self.n = batch, m, n
        # Used until a launch publishes its configuration; None takes the selector's.
        self._class_config = template_config or self._TEMPLATE_CONFIG
        self.program(self._template)

    def _template(self, cfg: dict | None) -> GEMMTemplate:
        """The template *cfg* pins, a description :meth:`forward` published; the class's
        tile, or the selector's, before one is."""
        pinned = dict(cfg or {})
        sms = pinned.pop("persistent_sms", None)
        if sms is None:
            sms = self._grid(self.batch, self.m, self.n)
        return GEMMTemplate(
            GEMMType.BATCHED,
            num_groups=self.batch,
            static_dims="mnk",
            sm_count=sms,
            device_index=self.device_index,
            config=pinned or self._class_config,
        )

    def _grid(self, batch: int, m: int, n: int) -> int:
        """Persistent CTAs: a full wave of the board's band tile, or every SM."""
        physical_sms = self.build_target.sm_count
        band = self._PERSISTENT_BANDS.get(self.build_target.calibration)
        if band is None:
            return physical_sms
        return self._full_wave_grid(self._tiles(band, batch, m, n), physical_sms)

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        b_nk = b.transpose(-2, -1)
        template = self.program(self._template)
        out = template(a, b_nk, out=out)
        if not self.config:
            # Publish the instantiation the launch ran, whole, so the next generation's
            # template compiles the same one.
            spec = template.spec_for(a, b_nk)
            self.config = {
                "block_m": spec.block_m,
                "block_n": spec.block_n,
                "block_k": spec.block_k,
                "num_stages": spec.num_stages,
                "num_math_wgs": spec.num_math_warpgroups,
                "pingpong": spec.pingpong,
                "epilogue_stage_n": spec.epilogue_stage_n,
                "epilogue_buffers": spec.epilogue_buffers,
                "swizzle_group_m": spec.swizzle_group_m,
                "a_box_rows": spec.a_box_rows,
                "persistent_sms": spec.num_sms,
            }
        return out


class _PingPongBand(NamedTuple):
    """Where :class:`BMMPingPongKernel` beats :class:`BMMPersistentKernel` on one calibrated
    board; its bounds are fitted there, as :class:`_PersistentBand`'s are."""

    # Waves of the persistent path's band tile the call fills at most. Over more, the
    # 128x128 tile's extra operand traffic per FLOP outweighs the epilogues it hides.
    max_waves: int
    # K range. Below it a tile has too few K blocks for one group's mainloop to cover
    # the other's epilogue; above it the mainloop decides, and the 128x256 tile reads
    # less per FLOP.
    min_k: int
    max_k: int


class BMMPingPongKernel(BMMPersistentKernel):
    """Persistent BMM on two ping-pong math warp-groups of 128x128 tiles, on a calibrated board.

    Each math warp-group runs whole tiles in turn, so one group's epilogue runs under the
    other's mainloop; :meth:`refusal` states the band where that outweighs the 128x128
    tile's extra operand traffic.
    """

    preferred_over = frozenset({"bmm_persistent"})
    _PINGPONG_BANDS = {"h200": _PingPongBand(max_waves=2, min_k=384, max_k=1024)}
    # Two staging chunks used in turn keep six ring stages; staging the whole tile leaves five.
    _TEMPLATE_CONFIG = {"block_m": 128, "block_n": 128, "pingpong": True, "epilogue_buffers": 2}

    @classmethod
    def entry_for(cls, call: BMMCall) -> Entry:
        """Build the ping-pong specialization for the call's K."""
        index = call.device_index
        identity = (call.batch, call.m, call.n, call.k, call.dtype, index)
        return identity, lambda: cls(call.batch, call.m, call.n, call.k, device_index=index)

    def __init__(self, batch: int, m: int, n: int, k: int, device_index: int | None = None) -> None:
        super().__init__(
            batch, m, n, device_index=device_index, template_config=self._template_config(k)
        )

    @classmethod
    def _template_config(cls, k: int) -> dict:
        """16-column staging chunks where K spans 16 or more 64-wide steps, else 32.

        Narrower chunks pay barriers per chunk that a shorter mainloop no longer hides.
        """
        width = 16 if -(-k // 64) >= 16 else 32
        return {**cls._TEMPLATE_CONFIG, "epilogue_stage_n": width}

    @classmethod
    def refusal(cls, call: BMMCall) -> str | None:
        band = cls._PINGPONG_BANDS.get(call.calibration)
        persistent = cls._PERSISTENT_BANDS.get(call.calibration)
        if band is None or persistent is None:
            return "has no ping-pong band for this board"
        if cls._tiles(persistent, call.batch, call.m, call.n) > band.max_waves * call.sm_count:
            return f"fills more than {band.max_waves} waves of the persistent tile"
        if not band.min_k <= call.k <= band.max_k:
            return f"requires k in [{band.min_k}, {band.max_k}], got k={call.k}"
        # Whole tiles leave through the asynchronous TMA store.
        block = cls._TEMPLATE_CONFIG["block_m"]
        if call.m % block or call.n % block:
            return f"requires m and n multiples of {block}, got m={call.m}, n={call.n}"
        # The grid has at most one CTA an SM, so past sm_count tiles some CTA runs a second
        # tile, whose mainloop covers the first tile's epilogue.
        if call.batch * (call.m // block) * (call.n // block) <= call.sm_count:
            return f"gives no CTA a second {block}x{block} tile on {call.sm_count} SMs"
        return super().refusal(call)

    def _grid(self, batch: int, m: int, n: int) -> int:
        """The power-of-two grid when it divides the 128x128 tiles, else every SM."""
        block = self._TEMPLATE_CONFIG["block_m"]
        tiles = batch * -(-m // block) * -(-n // block)
        return self._full_wave_grid(tiles, self.build_target.sm_count)


class _BMMFP8Kernel(Kernel, BMMFP8FwdInterface):
    """What the three batched FP8 programs share: the shape, the scales and the epilogue.

    Each subclass compiles one program in :meth:`_build_program` and states the config
    band that program runs at.
    """

    supported_archs: list[int] = [89, 90]

    @staticmethod
    def _operand_refusal(k: int, dtype: torch.dtype) -> str | None:
        """The programs read e4m3 operands in 32-wide FP8 WGMMA K steps."""
        if dtype != torch.float8_e4m3fn:
            return f"requires float8_e4m3fn operands, got {dtype}"
        if k % 32 == 0:
            return None
        return f"requires k a multiple of 32 (the FP8 WGMMA K step), got k={k}"

    @classmethod
    def refusal(cls, call: BMMFP8Call) -> str | None:
        return cls._operand_refusal(call.k, call.dtype)

    @classmethod
    def entry_for(cls, call: BMMFP8Call) -> Entry:
        """The device index is in the identity: the grid is sized from its SM count."""
        index = call.device_index
        arguments = (call.batch, call.m, call.n, call.k, call.dtype, call.out_dtype)
        return (*arguments, index), lambda: cls(*arguments, device_index=index)

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: dict | None = None,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        reason = self._operand_refusal(k, dtype)
        if reason is not None:
            raise ValueError(f"{type(self).__name__} {reason}")
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.sm_count = self.build_target.sm_count
        self._build_program()
        self.init_config(config)

    def _build_program(self) -> None:
        """Compile this class's program into ``self.kernel``."""
        raise NotImplementedError

    @classmethod
    def _sm90_tile_grid(cls, m: int, n: int, k: int, *, whole_tiles: bool) -> list[dict]:
        """The SM90 tile candidates under the shared-memory budget.

        Read by the persistent and the classic program, which tile the same body the
        same way and differ only in whether a tile may leave a tail.

        Args:
            m: Rows of one batch item's product.
            n: Columns of one batch item's product.
            k: Contraction dim.
            whole_tiles: Whether a tile must divide every extent, which the persistent
                grid needs because its body carries no epilogue guard.
        """
        # 27 KiB under the opt-in limit: the estimate below counts only the operand ring
        # and the output tile.
        budget = BLOCK_SHARED_BYTES_OPT_IN[90] - 27 * 1024
        configs = []
        # No 256-row tile: both programs return wrong products with one (eight warps over
        # 256 rows), whole tiles or not, so tuning must not be able to pick it.
        for block_m in (64, 128):
            if whole_tiles and m % block_m:
                continue
            for block_n in (64, 128, 256):
                if whole_tiles and n % block_n:
                    continue
                for block_k in (32, 64, 128):
                    if block_k > k or (whole_tiles and k % block_k):
                        continue
                    for num_stages in (2, 3, 4):
                        smem = (block_m * block_k + block_k * block_n) * num_stages
                        if smem + block_m * block_n * 2 > budget:
                            continue
                        configs.append(
                            {
                                "block_m": block_m,
                                "block_n": block_n,
                                "block_k": block_k,
                                "num_stages": num_stages,
                                "threads": 128 if block_m == 64 else 256,
                            }
                        )
        return configs

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        return self.program(self._compile)(a, b, scale_a, scale_b)

    def _compile(self, cfg: dict) -> Callable:
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)


class BMMFP8WSKernel(_BMMFP8Kernel):
    """Three-warpgroup warp-specialized persistent FP8 BMM.

    One producer warpgroup issues the TMA loads two consumer warpgroups run WGMMA over,
    so it claims the shapes whose whole tiles fill at least one and a half persistent
    waves; below that the specialization has nothing to hide behind.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"bmm_fp8_persistent"})

    # The tiles this kernel builds. The region, the default configuration and the
    # autotune filter read the same tuples, so widening one cannot leave them
    # disagreeing about what a tile can be. The region also admits ``block_m`` 256,
    # which no config picks: a shape that only fills a wave at that width is still
    # this kernel's, and its config falls back to 128.
    block_m_candidates: tuple[int, ...] = (128,)
    block_n_candidates: tuple[int, ...] = (128, 256)
    block_k_candidates: tuple[int, ...] = (64, 128)

    @classmethod
    def _k_tile_fits(cls, k: int, block_k: int) -> bool:
        """Whether *block_k* divides *k* into the two steps the mainloop pipelines over."""
        return block_k <= k and k % block_k == 0 and k // block_k >= 2

    @classmethod
    def refusal(cls, call: BMMFP8Call) -> "str | None":
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if not any(cls._k_tile_fits(call.k, tile) for tile in cls.block_k_candidates):
            return f"no K tile in {list(cls.block_k_candidates)} fits k={call.k}"
        # ceil(1.5 * sm_count): below one and a half persistent waves of whole tiles the
        # producer warpgroup has nothing left to hide behind.
        min_total_tiles = (call.sm_count * 3 + 1) // 2
        if not any(
            call.m % block_m == 0
            and call.n % block_n == 0
            and call.batch * (call.m // block_m) * (call.n // block_n) >= min_total_tiles
            for block_m in (*cls.block_m_candidates, 256)
            for block_n in cls.block_n_candidates
        ):
            return f"no whole-tile grid reaches {min_total_tiles} output tiles"
        return None

    def _build_program(self) -> None:
        self.kernel = _bmm_fp8_persistent_ws_kernel(
            self.batch, self.m, self.n, self.k, self.dtype_str, self.out_dtype_str, self.sm_count
        )

    @property
    def default_config(self) -> dict:
        return {
            "block_m": self.block_m_candidates[0],
            "block_n": 256 if self.n % 256 == 0 else 128,
            "block_k": 128 if self._k_tile_fits(self.k, 128) else 64,
            "num_stages": 3,
            "threads": 384,
            "group_size_m": 8,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = self.block_m_candidates[0]
        half_m = block_m // 2
        configs = []
        for block_n in self.block_n_candidates:
            if self.n % block_n:
                continue
            for block_k in self.block_k_candidates:
                if not self._k_tile_fits(self.k, block_k):
                    continue
                for num_stages in (2, 3, 4):
                    # fp8 operands are one byte; the two output tiles are bf16 / fp16.
                    mainloop = 2 * num_stages * half_m * block_k + num_stages * block_n * block_k
                    if mainloop + 2 * half_m * block_n * 2 > BLOCK_SHARED_BYTES_OPT_IN[90]:
                        continue
                    configs.extend(
                        {
                            "block_m": block_m,
                            "block_n": block_n,
                            "block_k": block_k,
                            "num_stages": num_stages,
                            "threads": 384,
                            "group_size_m": group_size_m,
                        }
                        for group_size_m in (1, 4, 8)
                    )
        return configs


class BMMFP8PersistentKernel(_BMMFP8Kernel):
    """Persistent FP8 BMM over a 128-aligned tile grid.

    A grid of one block per SM walking the output tiles, which removes the wave
    quantization the classic 3D grid pays; every extent must divide the tile, since the
    body carries no epilogue guard.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def refusal(cls, call: BMMFP8Call) -> "str | None":
        if call.m % 128 or call.n % 128 or call.k % 128:
            return f"requires m, n and k multiples of 128, got {call.m}, {call.n}, {call.k}"
        return super().refusal(call)

    def _build_program(self) -> None:
        self.kernel = _bmm_fp8_persistent_kernel(
            self.batch, self.m, self.n, self.k, self.dtype_str, self.out_dtype_str, self.sm_count
        )

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 3,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return self._sm90_tile_grid(self.m, self.n, self.k, whole_tiles=True)


class BMMFP8Kernel(_BMMFP8Kernel):
    """Classic 3D-grid FP8 BMM: one block per output tile, the batch on ``blockIdx.z``.

    Its epilogue guards the M and N tails, so it takes any shape an FP8 WGMMA K-step
    covers, on every FP8 tensor-core target.
    """

    general = True

    def _build_program(self) -> None:
        self.kernel = _bmm_fp8_kernel(
            self.batch, self.m, self.n, self.k, self.dtype_str, self.out_dtype_str
        )

    @property
    def default_config(self) -> dict:
        arch = self.build_target.arch
        if arch != 90:
            # Smaller tiles expose more parallelism while the grid fits in one wave.
            small_grid = self.batch * tilelang.cdiv(self.m, 64) * tilelang.cdiv(self.n, 64)
            tile = 64 if arch == 89 and small_grid <= self.sm_count else 128
            return {
                "block_m": tile,
                "block_n": tile,
                "block_k": 64 if self.k % 64 == 0 else 128,
                "num_stages": 2,
                "threads": 128,
            }
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 3,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        arch = self.build_target.arch
        if arch == 90:
            return self._sm90_tile_grid(self.m, self.n, self.k, whole_tiles=False)
        # A narrower sweep under the pre-SM90 per-block shared-memory limit; the copy
        # path zero-pads the K tail, so no candidate is filtered on k.
        budget = BLOCK_SHARED_BYTES_OPT_IN[arch]
        return [
            {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_stages": num_stages,
                "threads": 128,
            }
            for block_m in (64, 128)
            for block_n in (64, 128)
            for block_k in (64, 128)
            for num_stages in (2, 3)
            if (block_m * block_k + block_k * block_n) * num_stages + block_m * block_n * 2
            <= budget
        ]


class BMMFP8TransposeKernel(Kernel, BMMFP8TransposeFwdInterface):
    """Swap the last two axes of a contiguous FP8 ``[batch, rows, cols]`` tensor.

    Staged through shared memory so both the load and the store stay coalesced,
    which a strided element-wise copy cannot be at one byte per element.

    Data movement only: the result is bit-identical to
    ``src.transpose(-2, -1).contiguous()``.
    """

    # Square staging tile and the lanes that fill it. The thread count stays fixed during
    # autotune so a BMM tune does not spend most of its time on the copy kernel. TILE must
    # appear among the candidates.
    TILE: int = 64
    THREADS: int = 128
    TILE_CANDIDATES: tuple[int, ...] = (32, 64, 128)
    THREAD_CANDIDATES: tuple[int, ...] = (128,)

    @classmethod
    def entry_for(cls, call: BMMFP8TransposeCall) -> Entry:
        index = call.device_index
        arguments = (call.batch, call.rows, call.cols, call.dtype)
        return (*arguments, index), lambda: cls(*arguments, device_index=index)

    def __init__(
        self,
        batch: int,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        config: dict | None = None,
        *,
        device_index: int | None = None,
    ) -> None:
        """Build the transpose for one shape and dtype.

        Args:
            batch: Leading axis, untouched.
            rows: Extent of the source's second axis.
            cols: Extent of the source's third axis.
            dtype: Element dtype; ``torch.float8_e4m3fn`` is what the FP8 BMM passes.
            config: Optional tile override.
            device_index: CUDA device the kernel is built for.
        """
        super().__init__(device_index=device_index)
        self.dtype = dtype
        self.kernel = _bmm_fp8_transpose_kernel(batch, rows, cols, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"block": self.TILE, "threads": self.THREADS}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block": block, "threads": threads}
            for block in self.TILE_CANDIDATES
            for threads in self.THREAD_CANDIDATES
        ]

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Return ``src`` with its last two axes swapped, contiguous.

        Args:
            src: Contiguous $[B \\times rows \\times cols]$ tensor.

        Returns:
            A new contiguous $[B \\times cols \\times rows]$ tensor.
        """
        return self.program(self._compile)(src)

    def _compile(self, cfg: dict) -> Callable:
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)
