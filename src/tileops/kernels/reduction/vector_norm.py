"""Vector norm kernels (l1, l2, inf) using TileLang.

Computes vector norms along the last dimension:
  - l1: sum(|x|)
  - l2: sqrt(sum(x^2))
  - inf: max(|x|)

Operates on raw 2D (M, N) tensors; the kernel handles 256-element alignment
padding internally via masked loads with zero identity values.

Output dtype matches input dtype; internal computation in fp32.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    BlockConfigPlanner,
    align_up,
    device_smem_budget,
    restore_reduced,
    rows_for_axes,
    tune_by_forward,
)

__all__ = ["VectorNormKernel"]

_VECTOR_NORM_KINDS = {"l1", "l2", "inf"}


# Vector norm kernel


@functools.lru_cache(maxsize=32)
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
    _needs_pad = N_padded != N

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                transformed = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                if _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(0.0, "float32"),
                            )
                else:
                    # Optimization: fused load and cast - load directly to fp32 fragment
                    # This saves one intermediate buffer copy
                    # Need to guard M-dimension tail when M % block_m != 0
                    if M % block_m != 0:
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.if_then_else(
                                    pid_m * block_m + i < M,
                                    T.cast(x[pid_m * block_m + i, j], "float32"),
                                    T.cast(0.0, "float32"),
                                )
                    else:
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.cast(x[pid_m * block_m + i, j], "float32")

                if op_kind == "l1":
                    # l1 norm: sum(|x|)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            transformed[i, j] = T.abs(x_f32[i, j])
                    T.reduce_sum(transformed, acc, dim=1)
                elif op_kind == "l2":
                    # l2 norm: sqrt(sum(x^2))
                    # Optimization B: inline square computation to potentially reduce memory traffic
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            val = x_f32[i, j]
                            transformed[i, j] = val * val
                    T.reduce_sum(transformed, acc, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i] = T.sqrt(acc[i])
                else:
                    # inf norm: max(|x|)
                    # Note: T.reduce_max does not propagate NaN.
                    # NaN handling is done at the Op layer (InfNormFwdOp)
                    # by detecting NaN rows and patching the output.
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            transformed[i, j] = T.abs(x_f32[i, j])
                    T.reduce_max(transformed, acc, dim=1)

                # Cast back to output dtype
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(acc[i], dtype)

                # Write output
                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _vector_norm_kernel_tiled(M: int, N: int, op_kind: str, dtype: str, tile_n: int):
    """Build a tiled TileLang l1/l2/inf norm kernel.

    Iterates over the reduction dimension in chunks of ``tile_n`` columns,
    avoiding TileLang's single-fragment column limit at 32768 columns.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                transformed = T.alloc_fragment((block_m, tile_n), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                tile_acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                T.fill(acc, 0.0)

                for t in T.Serial(num_tiles):
                    for i in T.serial(block_m):
                        for j in T.Parallel(tile_n):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                                T.cast(
                                    x[pid_m * block_m + i, t * tile_n + j],
                                    "float32",
                                ),
                                T.cast(0.0, "float32"),
                            )

                    if op_kind == "l1":
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                transformed[i, j] = T.abs(x_f32[i, j])
                        T.reduce_sum(transformed, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]
                    elif op_kind == "l2":
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                transformed[i, j] = x_f32[i, j] * x_f32[i, j]
                        T.reduce_sum(transformed, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]
                    else:
                        # Note: T.reduce_max does not propagate NaN.
                        # NaN handling remains in the Op layer.
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                transformed[i, j] = T.abs(x_f32[i, j])
                        T.reduce_max(transformed, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.max(acc[i], tile_acc[i])

                if op_kind == "l2":
                    for i in T.Parallel(block_m):
                        acc[i] = T.sqrt(acc[i])

                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(acc[i], dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


# VectorNormKernel class


class VectorNormKernel(Kernel):
    """L1 / L2 / Inf norm forward kernel.

    Supports SM80+ architectures. Handles 256-element alignment padding inside
    the kernel. Computes norms via abs+sum (l1), square+sum+sqrt (l2), or
    abs+max (inf). Uses an N-tiled fallback for long rows that exceed
    TileLang's single-fragment column limit.

    Output dtype matches input dtype; internal computation in fp32.

    ``forward`` takes the tensor the op declares and reduces *reduce_axes* of it; the
    permute to rows and the shape of the result are this kernel's business.

    A row holding a NaN norms to NaN, matching ``torch.linalg.vector_norm``. The prim_func
    drops NaN values, so the ``inf`` kind patches those rows here — the compensation
    belongs to the implementation that needs it.

    Args:
        M: Rows the reduction leaves.
        N: Elements each row reduces.
        op_kind: One of "l1", "l2", "inf".
        dtype: Input data type (float32, float16, bfloat16).
        reduce_axes: Non-negative axis indices, ascending, that the reduction runs over.
        keepdim: Whether a reduced axis stays as a length-1 axis.
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
        device_index: CUDA device the input lives on, for the shared-memory budget.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        reduce_axes: "tuple[int, ...]",
        keepdim: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        if op_kind not in _VECTOR_NORM_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_VECTOR_NORM_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.reduce_axes = tuple(reduce_axes)
        self.keepdim = keepdim
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch.tensor([], dtype=dtype).element_size()
        self._smem_budget = device_smem_budget(device_index)
        self._planner = BlockConfigPlanner(
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
        )
        self._needs_tiling = self._planner.needs_tiling
        self.kernel = None
        if not self._needs_tiling:
            self.kernel = _vector_norm_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_to_str(self.dtype),
            )
        self.init_config(config, tune)
        if self._needs_tiling and not tune:
            bm = self.config.get("block_m", 1)
            threads = self.config.get("threads", DEFAULT_THREADS)
            if "tile_n" not in self.config or self.config["tile_n"] == 0:
                self.config["tile_n"] = self._planner.tile_n_for(bm, threads)
            reason = self._planner.reject_tile_n(bm, self.config["tile_n"], threads)
            if reason:
                raise ValueError(reason)

    @property
    def default_config(self) -> dict:
        return self._planner.default_config()

    @property
    def autotune_configs(self) -> list[dict]:
        return self._planner.autotune_configs()

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune vector norm, benchmarking tiled configs directly."""
        if not self._needs_tiling:
            return super().autotune(warmup=warmup, rep=rep)
        x = torch.randn(self.M, self.N, dtype=self.dtype, device=torch.cuda.current_device())
        tune_by_forward(self, x, warmup=warmup, rep=rep, forward=self._norm_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Norm *reduce_axes* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device.

        Returns:
            The normed tensor, same dtype as *x*.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        rows = rows_for_axes(x, self.reduce_axes)
        y = self._norm_rows(rows)
        if self.op_kind == "inf":
            nan_rows = rows.isnan().any(dim=-1)
            if nan_rows.any():
                y[nan_rows] = float("nan")
        return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)

    def _norm_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Norm the trailing axis of an ``(M, N)`` buffer."""
        dtype_str = self.dtype_to_str(self.dtype)
        if self._needs_tiling:
            program = _vector_norm_kernel_tiled(
                self.M, self.N, self.op_kind, dtype_str, self.config["tile_n"]
            )
        else:
            program = _vector_norm_kernel(self.M, self.N, self.op_kind, dtype_str)
        return program(self.config["block_m"], self.config["threads"])(x)
