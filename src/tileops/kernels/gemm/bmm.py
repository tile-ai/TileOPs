"""Batched GEMM kernel for BmmFwdOp.

Shapes are strict 3D-3D — ``a``: $[B \\times M \\times K]$, ``b``: $[B \\times K \\times N]$, ``c``: $[B \\times M \\times N]$.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.grouped_gemm.heuristics import GemmType
from tileops.kernels.grouped_gemm.template import GemmTemplate
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_sm_count, is_h200

from .call_spec import BmmCall

__all__ = [
    "BmmFp8Kernel",
    "BmmFp8TransposeKernel",
    "BmmKernel",
    "BmmTemplateKernel",
]


@functools.lru_cache(maxsize=64)
def _bmm_kernel(batch: int, m: int, n: int, k: int, dtype: str = "float16") -> Callable:
    """Pipelined batched GEMM for SM90.

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


def _(
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
    return torch.empty((batch, m, n), dtype=a.dtype, device=a.device)


class BmmKernel(Kernel):
    """Batched dense GEMM kernel (SM90).

    Computes ``C[b] = A[b] @ B[b]`` for ``b in [0, batch)`` where
    ``A: [batch, m, k]``, ``B: [batch, k, n]``, ``C: [batch, m, n]``.
    fp16 / bf16 inputs, fp32 accumulation. Grid maps the batch axis to
    ``blockIdx.z`` so all batches run in a single kernel launch.
    """

    supported_archs: list[int] = [90]
    general = True

    @classmethod
    def entry_for(cls, call: BmmCall) -> Entry:
        """Build the shape-specialized classic BMM fallback."""
        index = call.device.index if call.device is not None else None
        identity = (call.batch, call.m, call.n, call.k, call.dtype, call.tune, index)
        return identity, lambda: cls(
            call.batch,
            call.m,
            call.n,
            call.k,
            call.dtype,
            tune=call.tune,
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
        if k % 16 != 0:
            raise ValueError(
                f"BmmKernel requires contraction dim k to be a multiple of 16, got k={k}"
            )
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
        # Call the compiled JIT directly (cf. GemmKernel); the torch custom-op
        # is retained only for torch.compile compatibility.
        if not hasattr(self, "_compiled_kernel"):
            self._compiled_kernel = self.kernel(**self.config)
        return self._compiled_kernel(a, b)


class BmmTemplateKernel(Kernel):
    """Persistent H200 BMM adapter over :class:`GemmTemplate`.

    The template reads the zero-copy ``[batch, n, k]`` view of public
    ``b[batch, k, n]`` storage. :class:`BmmKernel` serves calls outside
    :meth:`applies`; this path takes its configuration from the template selector.
    """

    supported_archs: list[int] = [90]

    # The tile ``get_best_config`` picks for this path on H200. Selection and grid
    # sizing count the same tiles, or the region claimed is not the grid launched.
    TILE_M: int = 128
    TILE_N: int = 256

    # Half a persistent wave of those tiles is enough to beat BmmKernel. Fitted on
    # the manifest workloads: square-b16-512 reaches 128 tiles and wins,
    # square-b32-256 reaches 64 and loses. Re-fit against benchmarks/ops/bench_bmm.py
    # whenever the tile above or the epilogue changes.
    MIN_WAVE_DENOM: int = 2

    @classmethod
    def _tiles(cls, batch: int, m: int, n: int) -> int:
        """Output tiles this call launches at :attr:`TILE_M` x :attr:`TILE_N`."""
        return batch * -(-m // cls.TILE_M) * -(-n // cls.TILE_N)

    @classmethod
    def applies(cls, call: BmmCall) -> bool:
        step = 16 // call.dtype.itemsize
        tiles = cls._tiles(call.batch, call.m, call.n)
        return call.h200 and call.n % step == 0 and tiles * cls.MIN_WAVE_DENOM > call.sm_count

    @classmethod
    def _persistent_grid(cls, batch: int, m: int, n: int, physical_sms: int) -> int:
        """Choose a full-wave H200 grid for the selector's tile."""
        tiles = cls._tiles(batch, m, n)
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
        persistent_sms = (
            self._persistent_grid(batch, m, n, physical_sms)
            if is_h200(device_index)
            else physical_sms
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


class BmmFp8Kernel(Kernel):
    supported_archs: list[int] = [89, 90]

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
        if device is None:
            device = torch.device(torch.cuda.current_device())
        cc = torch.cuda.get_device_capability(device)
        if cc[0] < 9 and cc != (8, 9):
            # Fail fast with a clear message instead of a downstream
            # nvcc / PTX error at JIT time.
            raise NotImplementedError(
                f"BmmFp8Kernel requires FP8 tensor cores (sm89+); "
                f"got sm{cc[0]}{cc[1]} on the current device"
            )
        if k % 32 != 0:
            raise ValueError(
                f"BmmFp8Kernel requires contraction dim k to be a "
                f"multiple of 32 (FP8 WGMMA K-step), got k={k}"
            )
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        # Dispatch policy (in order of preference):
        #   1) 3-WG WS persistent (best throughput on aligned shapes;
        #      SM90 only — TMA + WGMMA);
        #   2) plain persistent (removes wave quantisation; SM90 only —
        #      the plain T.gemm body could run on pre-SM90 archs but is
        #      unvalidated there);
        #   3) classic 3D grid (handles arbitrary M/N tails; plain T.gemm,
        #      runs on any FP8 tensor-core target, sm89+).
        self._sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self._is_sm90 = cc[0] == 9
        self._use_ws = self._is_sm90 and self._ws_eligible(batch, m, n, k, self._sm_count)
        self._use_persistent = self._is_sm90 and (
            self._use_ws or self._persistent_eligible(m, n, k, self._use_ws)
        )
        if self._use_ws:
            self.kernel = _bmm_fp8_persistent_ws_kernel(
                batch, m, n, k, self.dtype_str, self.out_dtype_str, self._sm_count
            )
        elif self._use_persistent:
            self.kernel = _bmm_fp8_persistent_kernel(
                batch, m, n, k, self.dtype_str, self.out_dtype_str, self._sm_count
            )
        else:
            self.kernel = _bmm_fp8_kernel(batch, m, n, k, self.dtype_str, self.out_dtype_str)
        self.init_config(config, tune)

    # ---- Dispatch predicates ------------------------------------------
    @staticmethod
    def _ws_eligible(batch: int, m: int, n: int, k: int, sm_count: int) -> bool:
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
            bk_ok = (k % 64 == 0 and k // 64 >= 2) or (k % 128 == 0 and k // 128 >= 2)
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
        if not self._is_sm90:
            # Sized for the sm89 100KB per-block SMEM cap; K tails are
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
        if self._use_ws:
            SMEM_BUDGET_BYTES = 228 * 1024  # SM90 shared-memory cap
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
                            smem_main = 2 * ns * half_m * bk + ns * bn * bk  # fp8 = 1B
                            out_bytes = 2  # bf16/fp16 output
                            smem_c = 2 * half_m * bn * out_bytes
                            if smem_main + smem_c > SMEM_BUDGET_BYTES:
                                continue
                            for gsm in (1, 4, 8):
                                configs.append(
                                    {
                                        "block_m": bm,
                                        "block_n": bn,
                                        "block_k": bk,
                                        "num_stages": ns,
                                        "threads": 384,
                                        "group_size_m": gsm,
                                    }
                                )
            return configs

        if not self._is_sm90:
            # Classic 3D-grid sweep for the sm89 100KB cap.
            SMEM_BUDGET_BYTES = 100 * 1024
            configs = []
            for bm in (64, 128):
                for bn in (64, 128):
                    for bk in (64, 128):
                        for ns in (2, 3):
                            smem = (bm * bk + bk * bn) * ns + bm * bn * 2
                            if smem > SMEM_BUDGET_BYTES:
                                continue
                            configs.append(
                                {
                                    "block_m": bm,
                                    "block_n": bn,
                                    "block_k": bk,
                                    "num_stages": ns,
                                    "threads": 128,
                                }
                            )
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
                        raw_configs.append(
                            {
                                "block_m": bm,
                                "block_n": bn,
                                "block_k": bk,
                                "num_stages": ns,
                                "threads": threads,
                            }
                        )
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
                f"BmmFp8Kernel only supports torch.float8_e4m3fn, got {self.dtype}"
            )
        if not hasattr(self, "_compiled_kernel"):
            self._compiled_kernel = self.kernel(**self.config)
        return self._compiled_kernel(a, b, scale_a, scale_b)


class BmmFp8TransposeKernel(Kernel):
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

    def __init__(
        self,
        batch: int,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        """Build the transpose for one shape and dtype.

        Args:
            batch: Leading axis, untouched.
            rows: Extent of the source's second axis.
            cols: Extent of the source's third axis.
            dtype: Element dtype; ``torch.float8_e4m3fn`` is what the FP8 BMM passes.
            device: Device the kernel is built for.
            config: Optional tile override.
            tune: Whether to autotune the tile.
        """
        super().__init__(device_index=(device.index if isinstance(device, torch.device) else None))
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
