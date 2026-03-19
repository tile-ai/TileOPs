"""Vector norm kernels (l1, l2, inf) using TileLang.

Computes vector norms along the last dimension:
  - l1: sum(|x|)
  - l2: sqrt(sum(x^2))
  - inf: max(|x|)

Operates on 2D (M, N_padded) tensors; the Op layer handles reshape.
256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.

Output dtype matches input dtype; internal computation in fp32.
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

__all__ = ["VectorNormKernel"]

_VECTOR_NORM_KINDS = {"l1", "l2", "inf"}


# ---------------------------------------------------------------------------
# Vector norm kernel
# ---------------------------------------------------------------------------


def _vector_norm_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build a TileLang l1/l2/inf norm kernel.

    Computes vector norms along the last dimension:
      - l1: reduce_sum(|x|)
      - l2: sqrt(reduce_sum(x^2))
      - inf: reduce_max(|x|)

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Original hidden dimension (last dim, before padding).
        op_kind: One of "l1", "l2", "inf".
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
            out: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                transformed = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                # Load via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)

                # Cast to fp32
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                if op_kind == "l1":
                    # l1 norm: sum(|x|)
                    for i, j in T.Parallel(block_m, N_padded):
                        transformed[i, j] = T.abs(x_f32[i, j])
                    T.reduce_sum(transformed, acc, dim=1)
                elif op_kind == "l2":
                    # l2 norm: sqrt(sum(x^2))
                    for i, j in T.Parallel(block_m, N_padded):
                        transformed[i, j] = x_f32[i, j] * x_f32[i, j]
                    T.reduce_sum(transformed, acc, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i] = T.sqrt(acc[i])
                else:
                    # inf norm: max(|x|)
                    # Note: T.reduce_max does not propagate NaN.
                    # NaN handling is done at the Op layer (InfNormOp)
                    # by detecting NaN rows and patching the output.
                    for i, j in T.Parallel(block_m, N_padded):
                        transformed[i, j] = T.abs(x_f32[i, j])
                    T.reduce_max(transformed, acc, dim=1)

                # Cast back to output dtype
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(acc[i], dtype)

                # Write output
                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# custom_op wrappers for torch.compile compatibility
# ---------------------------------------------------------------------------


@torch.library.custom_op("top::vector_norm_fwd", mutates_args=())
def _vector_norm_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _vector_norm_kernel(M, N, op_kind, dtype_str)(block_m, threads)(x)


@_vector_norm_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# VectorNormKernel class
# ---------------------------------------------------------------------------


class VectorNormKernel(Kernel):
    """L1 / L2 / Inf norm forward kernel.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Computes norms via abs+sum (l1), square+sum+sqrt (l2),
    or abs+max (inf).

    Output dtype matches input dtype; internal computation in fp32.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "l1", "l2", "inf".
        dtype: Input data type (float32, float16, bfloat16).
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
        if op_kind not in _VECTOR_NORM_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_VECTOR_NORM_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self.kernel = _vector_norm_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_to_str(self.dtype),
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
        """Run the l1/l2/inf norm kernel.

        Args:
            x: Input tensor of shape (M, N_padded).

        Returns:
            Output tensor of shape (M,) with same dtype as input.
        """
        return _vector_norm_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_to_str(self.dtype),
            self.config["block_m"],
            self.config["threads"],
            x,
        )
