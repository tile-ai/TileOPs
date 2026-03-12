"""Cumulative scan kernels (cumsum, cumprod) using TileLang.

Implements an inclusive prefix scan along the last dimension using a
sequential loop (T.Serial) with a 1-D running accumulator to maintain the
correct data dependency chain without conflicting index patterns.

Both operate on 2D (M, N_padded) tensors; the Op layer handles reshape.
256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.
"""

import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    SHARED_MEMORY_BUDGET_BYTES,
    align_up,
)

__all__ = ["CumulativeKernel"]

# Identity elements for supported scan operations.
_SCAN_IDENTITY = {"sum": 0.0, "prod": 1.0}


def _cumulative_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build a TileLang inclusive prefix scan kernel.

    Uses a 1-D accumulator fragment ``(block_m,)`` that is updated
    sequentially across columns via ``T.Serial``.  This avoids
    accessing the same 2-D output buffer with two different index
    patterns (``[i, j-1]`` vs ``[i, j]``), which TileLang's
    structural-equality checker rejects.

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Hidden dimension (last dim, unpadded).
        op_kind: One of "sum", "prod".
        dtype: TileLang dtype string (e.g. "float16", "bfloat16", "float32").

    Returns:
        A TileLang JIT-compiled kernel factory accepting (block_m, threads).
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    identity = _SCAN_IDENTITY[op_kind]

    if op_kind == "sum":

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    out_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")

                    # Load input via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Cast to fp32 for numerical stability
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # Inclusive prefix sum using a 1-D accumulator.
                    T.fill(acc, identity)
                    for j in T.Serial(N_padded):
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + x_f32[i, j]
                            out_f32[i, j] = acc[i]

                    # Cast back to original dtype and write via shared memory
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = T.cast(out_f32[i, j], dtype)
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

    else:  # prod

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    out_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")

                    # Load input via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Cast to fp32 for numerical stability
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # Inclusive prefix product using a 1-D accumulator.
                    T.fill(acc, identity)
                    for j in T.Serial(N_padded):
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] * x_f32[i, j]
                            out_f32[i, j] = acc[i]

                    # Cast back to original dtype and write via shared memory
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = T.cast(out_f32[i, j], dtype)
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

    return _func


# ---------------------------------------------------------------------------
# custom_op wrappers for torch.compile compatibility
# ---------------------------------------------------------------------------


@torch.library.custom_op("top::cumulative_fwd", mutates_args=())
def _cumulative_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _cumulative_kernel(M, N, op_kind, dtype_str)(block_m, threads)(x)


@_cumulative_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, x):
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# CumulativeKernel class
# ---------------------------------------------------------------------------


class CumulativeKernel(Kernel):
    """Inclusive prefix scan kernel (cumsum / cumprod).

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Uses a sequential scan loop along the last dimension.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "sum", "prod".
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        if op_kind not in ("sum", "prod"):
            raise ValueError(f"Unsupported op_kind '{op_kind}'. Expected one of 'sum', 'prod'.")
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self.kernel = _cumulative_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget."""
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
        block_m = 1
        for bm in [1, 2, 4, 8]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
        block_ms = [bm for bm in [1, 2, 4, 8] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the cumulative scan kernel.

        Args:
            x: Input tensor of shape (M, N_padded).

        Returns:
            Output tensor of shape (M, N_padded).
        """
        return _cumulative_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )
