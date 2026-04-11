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

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up

__all__ = ["ArgreduceKernel"]

_ARGREDUCE_KINDS = {"argmax", "argmin"}


# ---------------------------------------------------------------------------
# Argreduce kernel
# ---------------------------------------------------------------------------


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
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                row_extreme = T.alloc_fragment((block_m,), "float32")
                out_idx = T.alloc_fragment((block_m,), "int64")

                # Load via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)

                # Cast to fp32
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                if op_kind == "argmax":
                    # For argmax: negate padded positions so they never win
                    # Padded positions are filled with -inf by the Op layer,
                    # so they already cannot win a max.
                    # Find row-wise max value using T.reduce_max
                    T.fill(row_extreme, -T.infinity("float32"))
                    T.reduce_max(x_f32, row_extreme, dim=1, clear=False)
                else:
                    # For argmin: padded positions are filled with +inf by Op layer,
                    # so they cannot win a min.
                    # Find row-wise min = -max(-x)
                    neg_x = T.alloc_fragment((block_m, N_padded), "float32")
                    for i, j in T.Parallel(block_m, N_padded):
                        neg_x[i, j] = -x_f32[i, j]
                    T.fill(row_extreme, -T.infinity("float32"))
                    T.reduce_max(neg_x, row_extreme, dim=1, clear=False)
                    # Negate back to get min value
                    for i in T.Parallel(block_m):
                        row_extreme[i] = -row_extreme[i]

                # Serial scan to find index of first occurrence matching extreme
                T.fill(out_idx, T.cast(0, "int64"))
                for i in T.Parallel(block_m):
                    for j in T.Serial(N):
                        if x_f32[i, j] == row_extreme[i]:
                            out_idx[i] = T.cast(j, "int64")
                            T.loop_break()

                # Write output
                T.copy(out_idx, out[pid_m * block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# custom_op wrappers for torch.compile compatibility
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ArgreduceKernel class
# ---------------------------------------------------------------------------


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

        Also enforces a TileLang layout-inference constraint:
        ``block_m * N_padded <= 2 * threads`` (each thread handles at most 2
        elements during the shared-memory copy).  This matters when the
        reduction dimension is small (e.g. dim=0 on a 3D tensor), causing
        ``N_padded`` to be much larger than ``N``.
        """
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m_smem = (48 * 1024) // smem_per_row
        threads = 128
        # TileLang layout constraint: block_m * N_padded <= 2 * threads
        max_block_m_layout = (2 * threads) // self.N_padded
        max_block_m = min(max_block_m_smem, max(max_block_m_layout, 1))
        block_m = 1
        for bm in [1, 2, 4, 8]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": threads}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m_smem = (48 * 1024) // smem_per_row
        threads_list = [128, 256]
        configs = []
        for threads in threads_list:
            # TileLang layout constraint: block_m * N_padded <= 2 * threads
            max_block_m_layout = (2 * threads) // self.N_padded
            max_block_m = min(max_block_m_smem, max(max_block_m_layout, 1))
            for bm in [1, 2, 4, 8]:
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
