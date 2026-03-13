"""Logical reduce kernels (any, all) using TileLang.

Casts input to bool (0/1 as float32), then reduces:
  - any: reduce_max (1 if any element is non-zero)
  - all: reduce_min (1 if all elements are non-zero)

Operates on 2D (M, N_padded) tensors; the Op layer handles reshape.
256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.

Output is always bool dtype.
"""

import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up

__all__ = ["LogicalReduceKernel"]

_LOGICAL_REDUCE_KINDS = {"any", "all"}

# TileLang does not support bool or complex dtypes as a storage dtype for
# T.alloc_shared / T.copy. When the caller's dtype is one of these, we use
# float32 internally instead. The Op layer is responsible for pre-converting
# the tensor to float32 before calling the kernel.
_FLOAT32_STORAGE_DTYPE = torch.float32
_UNSUPPORTED_STORAGE_DTYPES = frozenset(
    {
        torch.bool,
        torch.complex64,
        torch.complex128,
    }
)


# ---------------------------------------------------------------------------
# Logical reduce kernel
# ---------------------------------------------------------------------------


def _logical_reduce_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build a TileLang any/all kernel.

    Cast input to bool (0.0 or 1.0 in float32), then:
      - any: reduce_max over the row (1.0 if any element is non-zero)
      - all: reduce_min over the row (1.0 if all elements are non-zero)

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Original hidden dimension (last dim, before padding).
        op_kind: One of "any", "all".
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
            out: T.Tensor[(M,), "int8"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                bool_vals = T.alloc_fragment((block_m, N_padded), "float32")
                result = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), "int8")

                # Load via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)

                # Cast to fp32
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                # Cast to bool (0.0 or 1.0)
                for i, j in T.Parallel(block_m, N_padded):
                    bool_vals[i, j] = T.if_then_else(x_f32[i, j] != 0.0, 1.0, 0.0)

                if op_kind == "any":
                    # any: result is 1 if max(bool_vals) == 1
                    # Padding is 0 (neutral for OR/max)
                    T.reduce_max(bool_vals, result, dim=1)
                else:
                    # all: result is 1 if min(bool_vals) == 1
                    # Padding is 1 (neutral for AND/min)
                    T.reduce_min(bool_vals, result, dim=1)

                # Cast result to int8 (bool representation: 0 or 1)
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(result[i] > 0.5, "int8")

                # Write output
                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# custom_op wrappers for torch.compile compatibility
# ---------------------------------------------------------------------------


@torch.library.custom_op("top::logical_reduce_fwd", mutates_args=())
def _logical_reduce_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    out_int8 = _logical_reduce_kernel(M, N, op_kind, dtype_str)(block_m, threads)(x)
    return out_int8.bool()


@_logical_reduce_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, x):
    return torch.empty((M,), dtype=torch.bool, device=x.device)


# ---------------------------------------------------------------------------
# LogicalReduceKernel class
# ---------------------------------------------------------------------------


class LogicalReduceKernel(Kernel):
    """Any / all forward kernel.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Casts input to bool (0/1) and reduces via max (any) or
    min (all).

    Output dtype is always bool.

    Note: TileLang does not support bool or complex dtypes as a storage dtype
    for shared memory. When dtype is one of these, the kernel is compiled for
    float32 internally and the Op layer is responsible for pre-converting the
    input tensor to float32 before calling forward().

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "any", "all".
        dtype: Input data type (float32, float16, bfloat16, bool, complex64,
               or complex128).
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
        if op_kind not in _LOGICAL_REDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_LOGICAL_REDUCE_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        # TileLang cannot handle bool or complex dtypes as shared-memory
        # storage dtypes; remap all unsupported dtypes -> float32.
        self._kernel_dtype = (
            _FLOAT32_STORAGE_DTYPE if dtype in _UNSUPPORTED_STORAGE_DTYPES else dtype
        )
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self.kernel = _logical_reduce_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_to_str(self._kernel_dtype),
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget."""
        smem_per_row = self.N_padded * torch.tensor([], dtype=self._kernel_dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_m = 1
        for bm in [1, 2, 4, 8]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self._kernel_dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_ms = [bm for bm in [1, 2, 4, 8] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the any/all kernel.

        Args:
            x: Input tensor of shape (M, N_padded). Must have dtype matching
               _kernel_dtype (float32 when the original dtype is bool).

        Returns:
            Output tensor of shape (M,) with dtype bool.
        """
        return _logical_reduce_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_to_str(self._kernel_dtype),
            self.config["block_m"],
            self.config["threads"],
            x,
        )
