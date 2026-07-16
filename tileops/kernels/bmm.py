"""Batched GEMM kernel for BmmFwdOp.

Shapes are strict 3D-3D — ``a: [B, M, K]``, ``b: [B, K,N]``, ``c: [B, M, N]``.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = [
    "BmmFp8Kernel",
    "BmmKernel",
]  # from tileops.kernels.bmm import *


@functools.lru_cache(maxsize=64)
def _bmm_kernel(batch: int,
                m: int,
                n: int,
                k: int,
                dtype: str = "float16") -> Callable:
    """Pipelined batched GEMM for Hopper (SM90).

    Launches a 3D grid ``(ceildiv(n, block_n), ceildiv(m, block_m), batch)``.
    Each block loads its per-batch A/B tiles into SMEM through a ``T.Pipelined``
    K-loop and issues WGMMA into a fp32 accumulator; the epilogue guards the
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
    def _bmm_func(block_m: int = 128,
                  block_n: int = 128,
                  block_k: int = 64,
                  num_stages: int = 3,
                  threads: int = 128) -> Callable:

        @T.prim_func
        def _bmm_main(
                a: T.Tensor((batch, m, k), dtype),
                b: T.Tensor((batch, k, n), dtype),
                c: T.Tensor((batch, m, n), dtype),
        ) -> None:
            with T.Kernel(
                    T.ceildiv(n, block_n),
                    T.ceildiv(m, block_m),
                    batch,
                    threads=threads) as (bx, by, bz):
                a_smem = T.alloc_shared((block_m, block_k), dtype)
                b_smem = T.alloc_shared((block_k, block_n), dtype)
                c_smem = T.alloc_shared((block_m, block_n), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout({
                    a_smem: tilelang.layout.make_swizzled_layout(a_smem),
                    b_smem: tilelang.layout.make_swizzled_layout(b_smem),
                    c_smem: tilelang.layout.make_swizzled_layout(c_smem),
                })

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
                        a[bz, m_start:m_start + block_m, k_start:k_start + block_k],
                        a_smem,
                    )
                    T.copy(
                        b[bz, k_start:k_start + block_k, n_start:n_start + block_n],
                        b_smem,
                    )
                    T.gemm(a_smem, b_smem, c_local,
                           policy=T.GemmWarpPolicy.FullRow)

                # Epilogue: stage fp32 accum through SMEM before GMEM store.
                # 1-19% faster than direct fragment->GMEM — the M/N
                # tail guard breaks coalescing off the wgmma fragment layout,
                # but SMEM->GMEM stores stay coalesced under predication.
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
        k_exact = (k % block_k == 0)
        n_exact = (n % block_n == 0)
        m_exact = (m % block_m == 0)

        @T.prim_func
        def _bmm_fp8_main(
            a: T.Tensor((batch, m, k), dtype),  # type: ignore
            b: T.Tensor((batch, n, k), dtype),  # type: ignore  # N-major B
            scale_a: T.Tensor(scale_a_shape, "float32"),  # type: ignore
            scale_b: T.Tensor(scale_b_shape, "float32"),  # type: ignore
            c: T.Tensor((batch, m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(n, block_n),
                    T.ceildiv(m, block_m),
                    batch,
                    threads=threads) as (bx, by, bz):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                c_shared = T.alloc_shared((block_m, block_n), out_dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.annotate_layout({
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    c_shared: tilelang.layout.make_swizzled_layout(c_shared),
                })

                m_start = by * block_m
                n_start = bx * block_n
                T.clear(c_local)

                for kk in T.Pipelined(
                        T.ceildiv(k, block_k),
                        num_stages=num_stages):
                    k_start = kk * block_k
                    if k_exact and m_exact:
                        T.copy(
                            a[bz,
                              m_start:m_start + block_m,
                              k_start:k_start + block_k],
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
                            b[bz,
                              n_start:n_start + block_n,
                              k_start:k_start + block_k],
                            b_shared,
                        )
                    else:
                        for i, j in T.Parallel(block_n, block_k):
                            b_shared[i, j] = T.if_then_else(
                                (n_start + i < n) & (k_start + j < k),
                                b[bz, n_start + i, k_start + j],
                                T.cast(0, dtype),
                            )
                    T.gemm(a_shared, b_shared, c_local,
                           transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
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

                T.annotate_layout({
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    c_shared: tilelang.layout.make_swizzled_layout(c_shared),
                })

                for tile_b, tile_m, tile_n in T.Persistent(
                        [batch, T.ceildiv(m, block_m), T.ceildiv(n, block_n)],
                        wave_size=sm_count,
                        index=bx,
                        group_size=8):
                    m_start = tile_m * block_m
                    n_start = tile_n * block_n
                    T.clear(c_local)

                    for kk in T.Pipelined(
                            T.ceildiv(k, block_k),
                            num_stages=num_stages):
                        k_start = kk * block_k
                        T.copy(
                            a[tile_b,
                              m_start:m_start + block_m,
                              k_start:k_start + block_k],
                            a_shared,
                        )
                        T.copy(
                            b[tile_b,
                              n_start:n_start + block_n,
                              k_start:k_start + block_k],
                            b_shared,
                        )
                        T.gemm(a_shared, b_shared, c_local,
                               transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_m, block_n):
                        c_local[i, j] = (
                            c_local[i, j] * scale_a[0] * scale_b[0])
                    T.copy(c_local, c_shared)
                    T.copy(c_shared, c[tile_b,
                                       m_start:m_start + block_m,
                                       n_start:n_start + block_n])

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

                bs = T.alloc_local((1,), "int32")   # batch id
                ms = T.alloc_local((1,), "int32")   # M start (rows)
                ns_ = T.alloc_local((1,), "int32")  # N start (cols)

                ps0 = T.alloc_local((1,), "int32")
                ps1 = T.alloc_local((1,), "int32")

                swz_m = T.alloc_local((1,), "int32")
                swz_n = T.alloc_local((1,), "int32")

                T.annotate_layout({
                    A_smem_top: tilelang.layout.make_swizzled_layout(A_smem_top),
                    A_smem_bot: tilelang.layout.make_swizzled_layout(A_smem_bot),
                    B_smem: tilelang.layout.make_swizzled_layout(B_smem),
                    C_shared_wg0: tilelang.layout.make_swizzled_layout(C_shared_wg0),
                    C_shared_wg1: tilelang.layout.make_swizzled_layout(C_shared_wg1),
                })


                ab_full = T.alloc_barrier([128] * num_stages)
                ab_empty = T.alloc_barrier([256] * num_stages)

                gi_prod = T.alloc_var("int32", init=0)
                gi_cons_0 = T.alloc_var("int32", init=0)
                gi_cons_1 = T.alloc_var("int32", init=0)

                tx = T.get_thread_binding()

                # ═════════════════════════════════════════════════════════
                # Producer WG: tx < 128
                # ═════════════════════════════════════════════════════════
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
                                T.barrier_wait(
                                    ab_empty[slot],
                                    ((gi_prod // num_stages) & 1) ^ 1)
                                # Top half of A (rows ms..ms+half_m).
                                T.tma_copy(
                                    a[bs[0],
                                      ms[0]:ms[0] + half_m,
                                      k_start:k_start + block_k],
                                    A_smem_top[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                # Bottom half of A (rows ms+half_m..ms+block_m).
                                T.tma_copy(
                                    a[bs[0],
                                      ms[0] + half_m:ms[0] + block_m,
                                      k_start:k_start + block_k],
                                    A_smem_bot[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                # Full B tile shared between the two math WGs.
                                T.tma_copy(
                                    b[bs[0],
                                      ns_[0]:ns_[0] + block_n,
                                      k_start:k_start + block_k],
                                    B_smem[slot, :, :],
                                    barrier=ab_full[slot],
                                )
                                T.barrier_arrive(ab_full[slot])
                                gi_prod = gi_prod + 1

                # ═════════════════════════════════════════════════════════
                # Consumer WG0: 128 ≤ tx < 256 — top half (rows 0..half_m)
                # ═════════════════════════════════════════════════════════
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
                                T.barrier_wait(ab_full[slot],
                                               (gi_cons_0 // num_stages) & 1)
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
                                C_local_wg0[i, j] = (
                                    C_local_wg0[i, j]
                                    * scale_a[0] * scale_b[0])
                            T.copy(C_local_wg0, C_local_cast_wg0)

                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.copy(C_local_cast_wg0, C_shared_wg0)

                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.copy(C_shared_wg0,
                                   c[bs[0], ms[0], ns_[0]])

                # ═════════════════════════════════════════════════════════
                # Consumer WG1: tx ≥ 256 — bottom half (rows half_m..block_m)
                # ═════════════════════════════════════════════════════════
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
                                T.barrier_wait(ab_full[slot],
                                               (gi_cons_1 // num_stages) & 1)
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
                                C_local_wg1[i, j] = (
                                    C_local_wg1[i, j]
                                    * scale_a[0] * scale_b[0])
                            T.copy(C_local_wg1, C_local_cast_wg1)
                            # WG1's own named barrier (id=5) guards
                            # C_shared_wg1 reuse across waves.
                            T.sync_threads(barrier_id=5, arrive_count=128)
                            T.copy(C_local_cast_wg1, C_shared_wg1)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=5, arrive_count=128)
                            T.copy(C_shared_wg1,
                                   c[bs[0], ms[0] + half_m, ns_[0]])

        return _bmm_fp8_persistent_ws_main

    return _bmm_fp8_persistent_ws_func


@torch.library.custom_op("top::bmm_wrapped_kernel", mutates_args=())
def _bmm_wrapped_kernel(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Torch custom-op wrapper for ``torch.compile`` compatibility."""
    return _bmm_kernel(batch, m, n, k, dtype)(
        block_m, block_n, block_k, num_stages, threads)(a, b)


@_bmm_wrapped_kernel.register_fake
def _(batch: int, m: int, n: int, k: int, dtype: str, block_m: int, block_n: int,
      block_k: int, num_stages: int, threads: int,
      a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty((batch, m, n), dtype=a.dtype, device=a.device)


class BmmKernel(Kernel):
    """Batched dense GEMM kernel (SM90).

    Computes ``C[b] = A[b] @ B[b]`` for ``b in [0, batch)`` where
    ``A: [batch, m, k]``, ``B: [batch, k, n]``, ``C: [batch, m, n]``.
    fp16 / bf16 inputs, fp32 accumulation. Grid maps the batch axis to
    ``blockIdx.z`` so all batches run in a single kernel launch.
    """

    supported_archs: list[int] = [90]

    def __init__(self,
                 batch: int,
                 m: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        if k % 16 != 0:
            raise ValueError(
                f"BmmKernel requires contraction dim k to be a multiple of 16, got k={k}")
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.kernel = _bmm_kernel(batch, m, n, k, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # 64x64x64, stages=2, 1 warpgroup. Modal winner on H20-3e manifest
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
            {"block_m": bm, "block_n": bn, "block_k": bk,
             "num_stages": ns, "threads": 128}
            for bm in [64, 128]
            for bn in [64, 128]
            for bk in block_k_options
            for ns in [2, 3, 4]
        ]
        return [c for c in configs if self.k % c["block_k"] == 0]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Call the compiled JIT directly (cf. GemmKernel); the torch custom-op
        # is retained only for torch.compile compatibility.
        if not hasattr(self, "_compiled_kernel"):
            self._compiled_kernel = self.kernel(**self.config)
        return self._compiled_kernel(a, b)


class BmmFp8Kernel(Kernel):

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: Optional[torch.device] = None,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        # Every dispatched variant emits WGMMA + TMA (SM90+ only); fail fast
        # with a clear message on pre-Hopper GPUs instead of a downstream
        # nvcc / PTX error at JIT time.
        if device is None:
            device = torch.device(torch.cuda.current_device())
        cc = torch.cuda.get_device_capability(device)
        if cc[0] != 9:
            raise NotImplementedError(
                f"BmmFp8Kernel requires SM90 (Hopper); "
                f"got sm{cc[0]}{cc[1]} on the current device")
        if k % 32 != 0:
            raise ValueError(
                f"BmmFp8Kernel requires contraction dim k to be a "
                f"multiple of 32 (FP8 WGMMA K-step), got k={k}")
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        # Dispatch policy (in order of preference):
        #   1) 3-WG WS persistent (best throughput on aligned shapes);
        #   2) plain persistent (removes wave quantisation);
        #   3) classic 3D grid (handles arbitrary M/N tails).
        self._sm_count = torch.cuda.get_device_properties(
            device).multi_processor_count
        self._use_ws = self._ws_eligible(batch, m, n, k, self._sm_count)
        self._use_persistent = self._use_ws or self._persistent_eligible(
            m, n, k, self._use_ws)
        if self._use_ws:
            self.kernel = _bmm_fp8_persistent_ws_kernel(
                batch, m, n, k, self.dtype_str, self.out_dtype_str,
                self._sm_count)
        elif self._use_persistent:
            self.kernel = _bmm_fp8_persistent_kernel(
                batch, m, n, k, self.dtype_str, self.out_dtype_str,
                self._sm_count)
        else:
            self.kernel = _bmm_fp8_kernel(
                batch, m, n, k, self.dtype_str, self.out_dtype_str)
        self.init_config(config, tune)

    # ---- Dispatch predicates ------------------------------------------
    @staticmethod
    def _ws_eligible(batch: int, m: int, n: int, k: int,
                     sm_count: int) -> bool:
        min_total_tiles = (sm_count * 3 + 1) // 2  # ceil(1.5 * sm_count)
        for bm in (128, 256):
            if m % bm != 0 or (bm // 2) < 64:
                continue
            for bn in (128, 256):
                if n % bn != 0:
                    continue
                for bk in (64, 128):
                    if bk > k or k % bk != 0:
                        continue
                    if k // bk < 2:
                        continue
                    total_tiles = batch * (m // bm) * (n // bn)
                    if total_tiles < min_total_tiles:
                        continue
                    return True
        return False

    @staticmethod
    def _persistent_eligible(m: int, n: int, k: int, use_ws: bool) -> bool:
        if use_ws:
            bn_ok = (n % 128 == 0) or (n % 256 == 0)
            bk_ok = ((k % 64 == 0 and k // 64 >= 2)
                     or (k % 128 == 0 and k // 128 >= 2))
            return m % 128 == 0 and bn_ok and bk_ok
        return m % 128 == 0 and n % 128 == 0 and k % 128 == 0

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        if self._use_ws:
            bn = 256 if (self.n % 256 == 0) else 128
            bk = 128 if (self.k % 128 == 0 and self.k // 128 >= 2) else 64
            return {
                "block_m": 128,
                "block_n": bn,
                "block_k": bk,
                "num_stages": 3,
                "threads": 384,
                "group_size_m": 8,
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
        if self._use_ws:
            SMEM_BUDGET_BYTES = 228 * 1024  # H100/H20 shared-memory cap
            configs = []
            for bm in (128,):
                half_m = bm // 2
                for bn in (128, 256):
                    if self.n % bn != 0:
                        continue
                    for bk in (64, 128):
                        if bk > self.k or self.k % bk != 0:
                            continue
                        if self.k // bk < 2:
                            continue
                        for ns in (2, 3, 4):
                            smem_main = (
                                2 * ns * half_m * bk
                                + ns * bn * bk)  # fp8 = 1B
                            out_bytes = 2  # bf16/fp16 output
                            smem_c = 2 * half_m * bn * out_bytes
                            if smem_main + smem_c > SMEM_BUDGET_BYTES:
                                continue
                            for gsm in (1, 4, 8):
                                configs.append({
                                    "block_m": bm,
                                    "block_n": bn,
                                    "block_k": bk,
                                    "num_stages": ns,
                                    "threads": 384,
                                    "group_size_m": gsm,
                                })
            return configs

        SMEM_BUDGET_BYTES = 200 * 1024
        raw_configs = []
        for bm in (64, 128, 256):
            if self._use_persistent and self.m % bm != 0:
                continue
            for bn in (64, 128, 256):
                if self._use_persistent and self.n % bn != 0:
                    continue
                for bk in (32, 64, 128):
                    if bk > self.k:
                        continue
                    if self._use_persistent and self.k % bk != 0:
                        continue
                    threads = 128 if bm == 64 else 256
                    for ns in (2, 3, 4):
                        # Include c_shared (bm * bn * 2 bytes) in the budget
                        smem = (bm * bk + bk * bn) * ns + bm * bn * 2
                        if smem > SMEM_BUDGET_BYTES:
                            continue
                        raw_configs.append({
                            "block_m": bm,
                            "block_n": bn,
                            "block_k": bk,
                            "num_stages": ns,
                            "threads": threads,
                        })
        return raw_configs

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"BmmFp8Kernel only supports torch.float8_e4m3fn, "
                f"got {self.dtype}")
        if not hasattr(self, "_compiled_kernel"):
            self._compiled_kernel = self.kernel(**self.config)
        return self._compiled_kernel(a, b, scale_a, scale_b)
