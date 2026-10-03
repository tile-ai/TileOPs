"""Batched GEMM kernel for BmmFwdOp.

Shapes are strict 3D-3D — ``a``: $[B \\times M \\times K]$, ``b``: $[B \\times K \\times N]$, ``c``: $[B \\times M \\times N]$.
"""

import functools
from typing import Callable, NamedTuple, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops.kernels.gemm.call_spec import (
    BmmCall,
    BmmFp8Call,
    BmmFp8FwdInterface,
    BmmFp8TransposeCall,
    BmmFp8TransposeFwdInterface,
    BmmFwdInterface,
)
from tileops.kernels.gemm.persistent.heuristics import GemmType
from tileops.kernels.gemm.persistent.template import GemmTemplate
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import device_calibration, get_sm_count, get_sm_version

__all__ = [
    "BmmFp8Kernel",
    "BmmFp8PersistentKernel",
    "BmmFp8TransposeKernel",
    "BmmFp8WsKernel",
    "BmmKernel",
    "BmmPersistentKernel",
]


@functools.lru_cache(maxsize=64)
def _bmm_kernel(batch: int, m: int, n: int, k: int, dtype: str = "float16") -> Callable:
    """Pipelined batched GEMM.

    Launches a 3D grid ``(ceildiv(n, block_n), ceildiv(m, block_m), batch)``.
    Each block loads its per-batch A/B tiles into SMEM through a ``T.Pipelined``
    K-loop and issues WGMMA (MMA below SM90) into a fp32 accumulator; the epilogue guards the
    M/N tails so ``m``/``n`` need not be multiples of the block sizes.

    Args:
        batch: Number of independent GEMM problems (grid.z).
        m: Rows of each ``A[b]`` / ``C[b]``.
        n: Columns of each ``B[b]`` / ``C[b]``.
        k: Contraction dim shared across all batches.
        dtype: Activation / weight dtype string (``"float16"`` or
            ``"bfloat16"``).

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``(block_m, block_n,
        block_k, num_stages, threads)`` returns the compiled ``prim_func``.

    Note:
        WS pass is hard-disabled (``tl.disable_warp_specialized=True``);
        a full manifest scan on H20-3e showed it never won a shape.
    """
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _bmm_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        threads: int = 128,
    ) -> Callable:
        @T.prim_func
        def _bmm_main(
            a: T.Tensor((batch, m, k), dtype),
            b: T.Tensor((batch, k, n), dtype),
            c: T.Tensor((batch, m, n), dtype),
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

                # L2 rasterization: reshape the (bx, by) traversal order into
                # panels of ``panel_size`` blocks along N so neighbouring waves
                # reuse the same A/B rows in L2.
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


@functools.lru_cache(maxsize=32)
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
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _bmm_fp8_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        threads: int = 256,
    ) -> Callable:
        scale_a_shape = (1,)
        scale_b_shape = (1,)
        k_exact = k % block_k == 0
        n_exact = n % block_n == 0
        m_exact = m % block_m == 0

        @T.prim_func
        def _bmm_fp8_main(
            a: T.Tensor((batch, m, k), dtype),  # type: ignore
            b: T.Tensor((batch, n, k), dtype),  # type: ignore  # N-major B
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
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


@functools.lru_cache(maxsize=32)
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
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _bmm_fp8_persistent_func(
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 3,
        threads: int = 256,
    ) -> Callable:
        scale_a_shape = (1,)
        scale_b_shape = (1,)

        @T.prim_func
        def _bmm_fp8_persistent_main(
            a: T.Tensor((batch, m, k), dtype),  # type: ignore
            b: T.Tensor((batch, n, k), dtype),  # type: ignore  # N-major B
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
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


@functools.lru_cache(maxsize=32)
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
        assert threads == 384, (
            f"3-WG persistent WS BMM requires threads=384 "
            f"(1 producer + 2 consumer WGs); got threads={threads}"
        )
        assert block_m % 2 == 0 and block_m // 2 >= 64, (
            f"cooperative template needs block_m>=128 and even; got {block_m}"
        )
        assert m % block_m == 0 and n % block_n == 0 and k % block_k == 0, (
            f"WS variant requires aligned tile; got m={m}, n={n}, k={k}, "
            f"block=({block_m},{block_n},{block_k})"
        )
        half_m = block_m // 2
        num_pid_m = m // block_m
        num_pid_n = n // block_n
        k_iters = k // block_k
        assert k_iters >= 2, (
            f"WS variant needs k_iters>=2 for the 1-deep WGMMA pipeline; "
            f"got k={k}, block_k={block_k}, k_iters={k_iters}"
        )
        total_tiles = batch * num_pid_m * num_pid_n
        grid_cta = min(sm_count, total_tiles)
        max_waves = (total_tiles + grid_cta - 1) // grid_cta + 1

        scale_a_shape = (1,)
        scale_b_shape = (1,)

        @T.macro
        def _swizzle_decode(mn_id, swz_m, swz_n):
            _gin = T.int32(group_size_m * num_pid_n)
            _fpm = (mn_id // _gin) * T.int32(group_size_m)
            if T.int32(num_pid_m) - _fpm >= T.int32(group_size_m):
                swz_m[0] = _fpm + (mn_id % _gin) % T.int32(group_size_m)
                swz_n[0] = (mn_id % _gin) // T.int32(group_size_m)
            else:
                _gsm = T.int32(num_pid_m) - _fpm
                swz_m[0] = _fpm + (mn_id % _gin) % _gsm
                swz_n[0] = (mn_id % _gin) // _gsm

        @T.prim_func
        def _bmm_fp8_persistent_ws_main(
            a: T.Tensor((batch, m, k), dtype),  # type: ignore
            b: T.Tensor((batch, n, k), dtype),  # type: ignore  # N-major B
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
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
                            # BMM tile decode: flat_id = tile_b * (Mt*Nt) + mn_id
                            bs[0] = flat_id // T.int32(num_pid_m * num_pid_n)
                            mn_id = flat_id % T.int32(num_pid_m * num_pid_n)
                            _swizzle_decode(mn_id, swz_m, swz_n)
                            ms[0] = swz_m[0] * T.int32(block_m)
                            ns_[0] = swz_n[0] * T.int32(block_n)

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

                # Consumer WG0: 128 ≤ tx < 256 — top half (rows 0..half_m)
                elif tx < 256:
                    T.inc_max_nreg(240)

                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid_cta) * w + pid

                        if flat_id < T.int32(total_tiles):
                            bs[0] = flat_id // T.int32(num_pid_m * num_pid_n)
                            mn_id = flat_id % T.int32(num_pid_m * num_pid_n)
                            _swizzle_decode(mn_id, swz_m, swz_n)
                            ms[0] = swz_m[0] * T.int32(block_m)
                            ns_[0] = swz_n[0] * T.int32(block_n)

                            for kk in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_0 % num_stages
                                T.barrier_wait(ab_full[slot], (gi_cons_0 // num_stages) & 1)
                                T.wgmma_gemm(
                                    A_smem_top[slot, :, :],
                                    B_smem[slot, :, :],
                                    C_local_wg0,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(kk == 0),
                                )

                                if kk > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps0[0]])
                                ps0[0] = slot
                                gi_cons_0 = gi_cons_0 + 1

                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps0[0]])
                            T.warpgroup_fence_operand(C_local_wg0, num_regs=64)

                            for i, j in T.Parallel(half_m, block_n):
                                C_local_wg0[i, j] = C_local_wg0[i, j] * scale_a[0] * scale_b[0]
                            T.copy(C_local_wg0, C_local_cast_wg0)

                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.copy(C_local_cast_wg0, C_shared_wg0)

                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.copy(C_shared_wg0, c[bs[0], ms[0], ns_[0]])

                # Consumer WG1: tx ≥ 256 — bottom half (rows half_m..block_m)
                else:
                    T.inc_max_nreg(240)

                    for w in T.serial(max_waves):
                        flat_id = T.int32(grid_cta) * w + pid

                        if flat_id < T.int32(total_tiles):
                            bs[0] = flat_id // T.int32(num_pid_m * num_pid_n)
                            mn_id = flat_id % T.int32(num_pid_m * num_pid_n)
                            _swizzle_decode(mn_id, swz_m, swz_n)
                            ms[0] = swz_m[0] * T.int32(block_m)
                            ns_[0] = swz_n[0] * T.int32(block_n)

                            for kk in T.Pipelined(k_iters, num_stages=0):
                                slot = gi_cons_1 % num_stages
                                T.barrier_wait(ab_full[slot], (gi_cons_1 // num_stages) & 1)
                                T.wgmma_gemm(
                                    A_smem_bot[slot, :, :],
                                    B_smem[slot, :, :],
                                    C_local_wg1,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(kk == 0),
                                )
                                if kk > 0:
                                    T.wait_wgmma(1)
                                    T.barrier_arrive(ab_empty[ps1[0]])
                                ps1[0] = slot
                                gi_cons_1 = gi_cons_1 + 1

                            T.wait_wgmma(0)
                            T.barrier_arrive(ab_empty[ps1[0]])
                            T.warpgroup_fence_operand(C_local_wg1, num_regs=64)

                            # ── Epilogue: scale + cast + TMA-store ──
                            for i, j in T.Parallel(half_m, block_n):
                                C_local_wg1[i, j] = C_local_wg1[i, j] * scale_a[0] * scale_b[0]
                            T.copy(C_local_wg1, C_local_cast_wg1)
                            # WG1's own named barrier (id=5) guards
                            # C_shared_wg1 reuse across waves.
                            T.sync_threads(barrier_id=5, arrive_count=128)
                            T.copy(C_local_cast_wg1, C_shared_wg1)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=5, arrive_count=128)
                            T.copy(C_shared_wg1, c[bs[0], ms[0] + half_m, ns_[0]])

        return _bmm_fp8_persistent_ws_main

    return _bmm_fp8_persistent_ws_func


@functools.lru_cache(maxsize=32)
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

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3"])
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


class BmmKernel(Kernel, BmmFwdInterface):
    """Batched dense GEMM kernel.

    Computes ``C[b] = A[b] @ B[b]`` for ``b in [0, batch)`` where
    ``A: [batch, m, k]``, ``B: [batch, k, n]``, ``C: [batch, m, n]``.
    fp16 / bf16 inputs, fp32 accumulation. Grid maps the batch axis to
    ``blockIdx.z`` so all batches run in a single kernel launch.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general = True

    @classmethod
    def applies(cls, call: BmmCall) -> bool:
        return cls._region_refusal(call) is None

    @classmethod
    def refusal(cls, call: BmmCall) -> Optional[str]:
        return cls._region_refusal(call)

    @staticmethod
    def _region_refusal(call: BmmCall) -> Optional[str]:
        return BmmKernel._k_refusal(call.k)

    @staticmethod
    def _k_refusal(k: int) -> Optional[str]:
        """The tile loop steps K by 16."""
        return None if k % 16 == 0 else f"requires k a multiple of 16, got k={k}"

    @classmethod
    def entry_for(cls, call: BmmCall) -> Entry:
        """Build the shape-specialized classic BMM fallback."""
        index = call.device.index if call.device is not None else None
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
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        reason = self._k_refusal(k)
        if reason is not None:
            raise ValueError(f"BmmKernel {reason}")
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.kernel = _bmm_kernel(batch, m, n, k, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
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
        return [c for c in configs if self.k % c["block_k"] == 0]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_compiled_kernel"):
            self._compiled_kernel = self.kernel(**self.config)
        return self._compiled_kernel(a, b)


class _PersistentBand(NamedTuple):
    """Where the persistent path beats :class:`BmmKernel` on one calibrated board."""

    # The tile ``get_best_config`` picks for this path there. Selection and grid
    # sizing count the same tiles, or the region claimed is not the grid launched.
    tile_m: int
    tile_n: int
    # The path claims a call whose tiles fill 1 / min_wave_denom of a persistent wave.
    min_wave_denom: int


# Fitted on the manifest workloads: square-512 reaches 128 tiles and wins, square-256
# reaches 64 and loses. Re-fit against benchmarks/ops/bench_bmm.py whenever the tile or
# the epilogue changes. A board without an entry keeps BmmKernel.
_PERSISTENT_BANDS = {"h200": _PersistentBand(tile_m=128, tile_n=256, min_wave_denom=2)}


class BmmPersistentKernel(Kernel, BmmFwdInterface):
    """Persistent BMM adapter over :class:`GemmTemplate`, on a calibrated board.

    The template reads the zero-copy ``[batch, n, k]`` view of public
    ``b[batch, k, n]`` storage. :class:`BmmKernel` serves calls outside
    :meth:`applies`; this path takes its configuration from the template selector.
    """

    supported_archs: list[int] = [90]

    @staticmethod
    def _tiles(band: _PersistentBand, batch: int, m: int, n: int) -> int:
        """Output tiles this call launches at the band's tile."""
        return batch * -(-m // band.tile_m) * -(-n // band.tile_n)

    @classmethod
    def applies(cls, call: BmmCall) -> bool:
        return cls._region_refusal(call) is None

    @classmethod
    def refusal(cls, call: BmmCall) -> Optional[str]:
        return cls._region_refusal(call)

    @classmethod
    def _region_refusal(cls, call: BmmCall) -> Optional[str]:
        band = _PERSISTENT_BANDS.get(call.calibration)
        if band is None:
            return "has no persistent band for this board"
        # TMA addresses the K-contiguous operands and the output row in 16-byte steps.
        step = 16 // call.dtype.itemsize
        if call.n % step != 0 or call.k % step != 0:
            return f"requires n and k multiples of {step}, got n={call.n}, k={call.k}"
        if cls._tiles(band, call.batch, call.m, call.n) * band.min_wave_denom <= call.sm_count:
            return "does not fill a persistent wave"
        return None

    @classmethod
    def _persistent_grid(
        cls, band: _PersistentBand, batch: int, m: int, n: int, physical_sms: int
    ) -> int:
        """Choose a full-wave grid for the band's tile."""
        tiles = cls._tiles(band, batch, m, n)
        power_of_two_grid = 1 << (physical_sms.bit_length() - 1)
        return power_of_two_grid if tiles % power_of_two_grid == 0 else physical_sms

    @classmethod
    def entry_for(cls, call: BmmCall) -> Entry:
        """Build the persistent template specialization for this BMM call."""
        index = call.device.index if call.device is not None else None
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
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        physical_sms = get_sm_count(device_index)
        band = _PERSISTENT_BANDS.get(device_calibration(device_index))
        persistent_sms = (
            physical_sms if band is None else self._persistent_grid(band, batch, m, n, physical_sms)
        )
        self.template = GemmTemplate(
            GemmType.BATCHED,
            num_groups=batch,
            static_dims="mnk",
            sm_count=persistent_sms,
            device_index=device_index,
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b_nk = b.transpose(-2, -1)
        out = self.template(a, b_nk)
        if not self.config:
            spec = self.template.spec_for(a, b_nk)
            self.config = {
                "block_m": spec.block_m,
                "block_n": spec.block_n,
                "block_k": spec.block_k,
                "num_stages": spec.num_stages,
                "num_math_wgs": spec.num_math_warpgroups,
                "epilogue_stage_n": spec.epilogue_stage_n,
                "persistent_sms": spec.num_sms,
            }
        return out


class _BmmFp8Kernel(Kernel, BmmFp8FwdInterface):
    """What the three batched FP8 programs share: the shape, the scales and the epilogue.

    Each subclass compiles one program in :meth:`_build_program` and states the config
    band that program runs at.
    """

    supported_archs: list[int] = [89, 90]

    @staticmethod
    def _k_refusal(k: int) -> Optional[str]:
        """The FP8 WGMMA K-step is 32 elements wide."""
        if k % 32 == 0:
            return None
        return f"requires k a multiple of 32 (the FP8 WGMMA K step), got k={k}"

    @classmethod
    def applies(cls, call: BmmFp8Call) -> bool:
        return cls._k_refusal(call.k) is None

    @classmethod
    def refusal(cls, call: BmmFp8Call) -> Optional[str]:
        """The K-step reason where that is what refuses, else what the region says."""
        return cls._k_refusal(call.k) or (None if cls.applies(call) else "does not serve this call")

    @classmethod
    def entry_for(cls, call: BmmFp8Call) -> Entry:
        """The device index is in the identity: the grid is sized from its SM count."""
        index = call.device.index if call.device is not None else None
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
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        reason = self._k_refusal(k)
        if reason is not None:
            raise ValueError(f"{type(self).__name__} {reason}")
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.sm_count = get_sm_count(self.device_index)
        self._build_program()
        self.init_config(config, tune)

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
        budget = 200 * 1024
        configs = []
        for block_m in (64, 128, 256):
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
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"{type(self).__name__} only supports torch.float8_e4m3fn, got {self.dtype}"
            )
        if not hasattr(self, "_compiled_kernel"):
            self._compiled_kernel = self.kernel(**self.config)
        return self._compiled_kernel(a, b, scale_a, scale_b)


class BmmFp8WsKernel(_BmmFp8Kernel):
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
    def applies(cls, call: BmmFp8Call) -> bool:
        if cls._k_refusal(call.k) is not None:
            return False
        if not any(cls._k_tile_fits(call.k, tile) for tile in cls.block_k_candidates):
            return False
        # ceil(1.5 * sm_count): below one and a half persistent waves of whole tiles the
        # producer warpgroup has nothing left to hide behind.
        min_total_tiles = (call.sm_count * 3 + 1) // 2
        return any(
            call.m % block_m == 0
            and call.n % block_n == 0
            and call.batch * (call.m // block_m) * (call.n // block_n) >= min_total_tiles
            for block_m in (*cls.block_m_candidates, 256)
            for block_n in cls.block_n_candidates
        )

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


class BmmFp8PersistentKernel(_BmmFp8Kernel):
    """Persistent FP8 BMM over a 128-aligned tile grid.

    A grid of one block per SM walking the output tiles, which removes the wave
    quantization the classic 3D grid pays; every extent must divide the tile, since the
    body carries no epilogue guard.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call: BmmFp8Call) -> bool:
        return (
            cls._k_refusal(call.k) is None
            and call.m % 128 == 0
            and call.n % 128 == 0
            and call.k % 128 == 0
        )

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


class BmmFp8Kernel(_BmmFp8Kernel):
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
    def _arch(self) -> int:
        """The architecture this instance compiled for, which sizes its shared memory."""
        return get_sm_version(self.device_index)

    @property
    def default_config(self) -> dict:
        if self._arch != 90:
            # Sized for the sm89 per-block shared-memory limit; K tails are
            # zero-padded by the classic copy path.
            return {
                "block_m": 128,
                "block_n": 128,
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
        if self._arch == 90:
            return self._sm90_tile_grid(self.m, self.n, self.k, whole_tiles=False)
        # A narrower sweep under the pre-SM90 per-block shared-memory limit; the copy
        # path zero-pads the K tail, so no candidate is filtered on k.
        budget = BLOCK_SHARED_BYTES_OPT_IN[self._arch]
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


class BmmFp8TransposeKernel(Kernel, BmmFp8TransposeFwdInterface):
    """Swap the last two axes of a contiguous FP8 ``[batch, rows, cols]`` tensor.

    Staged through shared memory so both the load and the store stay coalesced,
    which a strided element-wise copy cannot be at one byte per element.

    Data movement only: the result is bit-identical to
    ``src.transpose(-2, -1).contiguous()``.
    """

    # Square staging tile and the lanes that fill it. Fitted on the FP8 BMM
    # workloads in benchmarks/ops/bench_bmm.py; re-fit against those when the
    # staging layout changes. Keep the thread count fixed during autotune so a
    # BMM tune does not spend most of its time on the copy kernel. TILE must
    # appear among the candidates.
    TILE: int = 64
    THREADS: int = 128
    TILE_CANDIDATES: tuple[int, ...] = (32, 64, 128)
    THREAD_CANDIDATES: tuple[int, ...] = (128,)

    @classmethod
    def entry_for(cls, call: BmmFp8TransposeCall) -> Entry:
        index = call.device.index if call.device is not None else None
        arguments = (call.batch, call.rows, call.cols, call.dtype)
        return (*arguments, index), lambda: cls(*arguments, device_index=index)

    def __init__(
        self,
        batch: int,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        """Build the transpose for one shape and dtype.

        Args:
            batch: Leading axis, untouched.
            rows: Extent of the source's second axis.
            cols: Extent of the source's third axis.
            dtype: Element dtype; ``torch.float8_e4m3fn`` is what the FP8 BMM passes.
            config: Optional tile override.
            tune: Whether to autotune the tile.
            device_index: CUDA device the kernel is built for.
        """
        super().__init__(device_index=device_index)
        self.dtype = dtype
        self.kernel = _bmm_fp8_transpose_kernel(batch, rows, cols, self.dtype_str)
        self.init_config(config, tune)

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
        if not hasattr(self, "_compiled_kernel"):
            self._compiled_kernel = self.kernel(**self.config)
        return self._compiled_kernel(src)
