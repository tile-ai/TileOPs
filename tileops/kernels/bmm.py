"""Batched GEMM kernel for BmmFwdOp.

Shapes are strict 3D-3D — ``a: [B, M, K]``, ``b: [B, K,N]``, ``c: [B, M, N]``.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["BmmKernel"] # from tileops.kernels.bmm import *


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
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((batch, m, n), dtype=inputs[0].dtype, device=inputs[0].device)


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
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 2,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": bm, "block_n": bn, "block_k": bk,
             "num_stages": ns, "threads": 128}
            for bm in [64, 128]
            for bn in [64, 128]
            for bk in [32, 64]
            for ns in [2, 3, 4]
        ]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Call the compiled JIT directly (cf. GemmKernel); the torch custom-op
        # is retained only for torch.compile compatibility.
        return self.kernel(**self.config)(a, b)
