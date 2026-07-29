"""Argreduce kernels (argmax, argmin) using TileLang.

Implements a two-step kernel: first finds the extreme value via parallel reduce,
then scans for the first index matching that value.
Operates on 2D (M, N_padded) tensors; the Op layer handles reshape.
256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.

Output is always int64 (index values).
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    SHARED_MEMORY_BUDGET_BYTES,
    align_up,
)

__all__ = ["ArgreduceKernel"]

_ARGREDUCE_KINDS = {"argmax", "argmin"}


# Argreduce kernel


@functools.lru_cache(maxsize=32)
def _argreduce_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build a TileLang argmax/argmin kernel.

    Uses a two-step approach:
      Step 1: Load data, cast to fp32, find row-wise max/min using T.reduce_max.
      Step 2: Serial scan to find the index of the first occurrence of
              the max/min value.

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Original hidden dimension (last dim, before padding).
        op_kind: One of "argmax", "argmin".
        dtype: TileLang dtype string (e.g. "float16", "bfloat16", "float32").

    Returns:
        A TileLang JIT-compiled kernel factory accepting (block_m, threads).
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            out: T.Tensor[(M,), "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                # Key improvement: Don't materialize full x_f32 tensor!
                # Only allocate accumulators for reduction state
                row_extreme = T.alloc_fragment((block_m,), "float32")
                out_idx = T.alloc_fragment((block_m,), "int64")

                # Load via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)

                # Single-pass pair reduction: find value and index simultaneously
                # Initialize accumulators
                if op_kind == "argmax":
                    T.fill(row_extreme, -T.infinity("float32"))
                else:
                    T.fill(row_extreme, T.infinity("float32"))
                T.fill(out_idx, T.cast(0, "int64"))

                # Stream through data once, maintaining (value, index) pair
                # NOTE: Still using Serial scan due to TileLang limitations:
                # - T.atomic_min() fails (register copy bug)
                # - No T.reduce_with_index() primitive
                # - No warp shuffle primitives
                # But at least we avoid materializing full N_padded×block_m tensor
                for i in T.Parallel(block_m):
                    for j in T.Serial(N):
                        # Cast on-the-fly, don't store
                        val_f32 = T.cast(shared_buf[i, j], "float32")

                        # Pair reduction combine logic
                        if op_kind == "argmax":
                            should_update = (val_f32 > row_extreme[i]) or \
                                          (val_f32 == row_extreme[i] and j < out_idx[i])
                        else:  # argmin
                            should_update = (val_f32 < row_extreme[i]) or \
                                          (val_f32 == row_extreme[i] and j < out_idx[i])

                        if should_update:
                            row_extreme[i] = val_f32
                            out_idx[i] = T.cast(j, "int64")

                # Write output
                T.copy(out_idx, out[pid_m * block_m])

        return main

    return _func


# custom_op wrappers for torch.compile compatibility


@torch.library.custom_op("top::argreduce_fwd", mutates_args=())
def _argreduce_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _argreduce_kernel(M, N, op_kind, dtype_str)(block_m, threads)(x)


@_argreduce_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, x):
    return torch.empty((M,), dtype=torch.int64, device=x.device)


# ArgreduceKernel class


class ArgreduceKernel(Kernel):
    """Argmax / argmin forward kernel.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Implements a two-step approach: parallel reduce to find
    the extreme value, then serial scan to find the first matching index.

    Output dtype is always int64.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "argmax", "argmin".
        dtype: Input data type (float32, float16, or bfloat16).
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
        if op_kind not in _ARGREDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_ARGREDUCE_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self.kernel = _argreduce_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget.

        When the original reduction dimension *N* is smaller than the
        alignment boundary (``DEFAULT_ALIGNMENT``), padding inflates the row
        width and TileLang's copy-layout inference requires
        ``block_m * N_padded <= 2 * threads``.  For large *N* (>= alignment)
        the rows are dense and the layout works at any block_m that fits in
        shared memory, so the constraint is skipped.
        """
        if self.N_padded == 0:
            raise ValueError(
                "Reduction dimension is empty (N=0). "
                "argmax/argmin over an empty dimension is undefined."
            )
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m_smem = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
        if max_block_m_smem == 0:
            raise ValueError(
                f"A single row requires {smem_per_row} bytes of shared memory, "
                f"which exceeds the {SHARED_MEMORY_BUDGET_BYTES}-byte budget "
                f"(N_padded={self.N_padded}, dtype={self.dtype}). "
                f"Reduce the reduction dimension or use a dtype with smaller element size."
            )
        threads = 128
        max_block_m = max_block_m_smem
        if self.N < DEFAULT_ALIGNMENT:
            # TileLang layout constraint: only needed when heavy padding
            max_block_m_layout = (2 * threads) // self.N_padded
            max_block_m = min(max_block_m_smem, max(max_block_m_layout, 1))
        block_m = 1
        for bm in [1, 2, 4, 8]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": threads}

    @property
    def autotune_configs(self) -> list[dict]:
        if self.N_padded == 0:
            raise ValueError(
                "Reduction dimension is empty (N=0). "
                "argmax/argmin over an empty dimension is undefined."
            )
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m_smem = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
        if max_block_m_smem == 0:
            raise ValueError(
                f"A single row requires {smem_per_row} bytes of shared memory, "
                f"which exceeds the {SHARED_MEMORY_BUDGET_BYTES}-byte budget "
                f"(N_padded={self.N_padded}, dtype={self.dtype}). "
                f"Reduce the reduction dimension or use a dtype with smaller element size."
            )
        # Extended thread count and block_m options for better tuning
        threads_list = [128, 256, 512]
        configs = []
        for threads in threads_list:
            max_block_m = max_block_m_smem
            if self.N < DEFAULT_ALIGNMENT:
                # TileLang layout constraint: only needed when heavy padding
                max_block_m_layout = (2 * threads) // self.N_padded
                max_block_m = min(max_block_m_smem, max(max_block_m_layout, 1))
            # Expand block_m options to include 16 and 32 for better coalescing
            for bm in [1, 2, 4, 8, 16, 32]:
                if bm <= max_block_m:
                    configs.append({"block_m": bm, "threads": threads})
        return configs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the argmax/argmin kernel.

        Args:
            x: Input tensor of shape (M, N_padded).

        Returns:
            Output tensor of shape (M,) with dtype int64.
        """
        return _argreduce_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )
