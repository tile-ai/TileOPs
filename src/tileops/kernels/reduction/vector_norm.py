"""Vector norm kernels (l1, l2, inf) using TileLang.

Computes vector norms along the last dimension:
  - l1: sum(|x|)
  - l2: sqrt(sum(x^2))
  - inf: max(|x|), reduced over IEEE bit patterns as int32 so a row holding NaN
    norms to NaN without a second look at the input

Operates on raw 2D (M, N) tensors; the kernel handles 256-element alignment
padding internally via masked loads with zero identity values.

Output dtype matches input dtype; l1 and l2 compute in fp32.
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
    edge_axis_plan,
    edge_axis_split,
    reduce_down_rows,
    restore_reduced,
    rows_for_axes,
    tune_by_forward,
)

__all__ = ["VectorNormKernel"]

_VECTOR_NORM_KINDS = {"l1", "l2", "inf"}


# Vector norm kernel


# Dtype of the fragment each kind reduces. ``inf`` reduces IEEE bit patterns as int32:
# ``T.abs`` clears the sign bit, non-negative patterns sort like their floats, and every
# NaN pattern outranks +inf's, so one integer max yields the norm and reports NaN.
_WORK_DTYPE = {"l1": "float32", "l2": "float32", "inf": "int32"}


@T.macro
def _prepared(x, row, col, in_bounds, op_kind: str):
    """One element, transformed into what *op_kind* reduces, or its identity."""
    value = T.if_then_else(in_bounds, T.cast(x[row, col], "float32"), T.cast(0.0, "float32"))
    if op_kind == "l1":
        return T.abs(value)
    if op_kind == "l2":
        return value * value
    return T.reinterpret(T.abs(value), "int32")


@T.macro
def _finished(accumulated, op_kind: str, dtype: str):
    """The accumulator, in the output dtype."""
    if op_kind == "inf":
        return T.cast(T.reinterpret(accumulated, "float32"), dtype)
    return T.cast(accumulated, dtype)


@functools.lru_cache(maxsize=32)
def _vector_norm_kernel(M: int, N: int, op_kind: str, dtype: str, partial: bool = False):
    """Build a TileLang l1/l2/inf norm kernel.

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Original hidden dimension (last dim, before padding).
        op_kind: One of "l1", "l2", "inf".
        dtype: TileLang dtype string (e.g. "float16", "bfloat16", "float32").
        partial: Write fp32 partials for an outer pass — no l2 sqrt, no cast
            to the storage dtype. ``inf`` partials stay NaN-carrying values.

    Returns:
        A TileLang JIT-compiled kernel factory accepting (block_m, threads).
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    work_dtype = _WORK_DTYPE[op_kind]
    out_dtype = "float32" if partial else dtype

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                work = T.alloc_fragment((block_m, N_padded), work_dtype)
                acc = T.alloc_fragment((block_m,), work_dtype)
                out_local = T.alloc_fragment((block_m,), out_dtype)

                # Loaded straight into the working fragment, with no staging buffer.
                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        if _needs_pad:
                            in_bounds = T.And(pid_m * block_m + i < M, j < N)
                        else:
                            in_bounds = pid_m * block_m + i < M
                        work[i, j] = _prepared(x, pid_m * block_m + i, j, in_bounds, op_kind)

                if op_kind == "l1":
                    T.reduce_sum(work, acc, dim=1)
                elif op_kind == "l2":
                    T.reduce_sum(work, acc, dim=1)
                    if not partial:
                        for i in T.Parallel(block_m):
                            acc[i] = T.sqrt(acc[i])
                else:
                    T.reduce_max(work, acc, dim=1)

                for i in T.Parallel(block_m):
                    out_local[i] = _finished(acc[i], op_kind, out_dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _vector_norm_kernel_tiled(
    M: int, N: int, op_kind: str, dtype: str, tile_n: int, partial: bool = False
):
    """Build a tiled TileLang l1/l2/inf norm kernel.

    Iterates over the reduction dimension in chunks of ``tile_n`` columns,
    avoiding TileLang's single-fragment column limit at 32768 columns.
    ``partial`` writes fp32 partials for an outer pass, as in
    ``_vector_norm_kernel``.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    work_dtype = _WORK_DTYPE[op_kind]
    out_dtype = "float32" if partial else dtype

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                work = T.alloc_fragment((block_m, tile_n), work_dtype)
                acc = T.alloc_fragment((block_m,), work_dtype)
                tile_acc = T.alloc_fragment((block_m,), work_dtype)
                out_local = T.alloc_fragment((block_m,), out_dtype)

                # Zero is every kind's identity; for inf it is +0.0's bit pattern.
                T.fill(acc, 0)

                for t in T.Serial(num_tiles):
                    for i in T.serial(block_m):
                        for j in T.Parallel(tile_n):
                            work[i, j] = _prepared(
                                x,
                                pid_m * block_m + i,
                                t * tile_n + j,
                                T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                                op_kind,
                            )

                    if op_kind == "inf":
                        T.reduce_max(work, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.max(acc[i], tile_acc[i])
                    else:
                        T.reduce_sum(work, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]

                if op_kind == "l2" and not partial:
                    for i in T.Parallel(block_m):
                        acc[i] = T.sqrt(acc[i])

                for i in T.Parallel(block_m):
                    out_local[i] = _finished(acc[i], op_kind, out_dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _inf_merge_kernel(A: int, B: int, out_dtype: str, threads: int):
    """Merge fp32 abs-max partials down the lead axis over IEEE bit patterns.

    A plain float max drops NaN; comparing the partials' non-negative bit
    patterns as int32 keeps it, exactly like the rows pass. One thread owns a
    kept column, so every step of the walk is one coalesced pass.
    """

    @tilelang.jit(out_idx=[1])
    def _func():
        @T.prim_func
        def main(
            partials: T.Tensor[(A, B), "float32"],  # noqa: F821
            out: T.Tensor[(B,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(B, threads), threads=threads) as pid_b:
                tx = T.get_thread_binding()
                acc = T.alloc_local((1,), "int32")

                acc[0] = 0
                with T.If(pid_b * threads + tx < B):  # noqa: SIM117
                    with T.Then():
                        for a in T.serial(A):
                            acc[0] = T.max(
                                acc[0],
                                T.reinterpret(partials[a, pid_b * threads + tx], "int32"),
                            )
                        out[pid_b * threads + tx] = T.cast(
                            T.reinterpret(acc[0], "float32"), out_dtype
                        )

        return main

    return _func


# VectorNormKernel class


class VectorNormKernel(Kernel):
    """L1 / L2 / Inf norm forward kernel.

    Supports SM80+ architectures. Handles 256-element alignment padding inside
    the kernel. Computes norms via abs+sum (l1), square+sum+sqrt (l2), or
    abs+max (inf). Uses an N-tiled fallback for long rows that exceed
    TileLang's single-fragment column limit.

    Output dtype matches input dtype. l1 and l2 accumulate in fp32; ``inf`` reduces
    int32 bit patterns, which is what carries NaN.

    ``forward`` takes the tensor the op declares and reduces *reduce_axes* of it; the
    permute to rows and the shape of the result are this kernel's business.

    A row holding a NaN norms to NaN, matching ``torch.linalg.vector_norm``. The ``inf``
    kind gets that from its reducer, so no pass over the input beyond the reduction
    itself decides it.

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
        k, j = edge_axis_split(x.ndim, self.reduce_axes)
        if k:
            y = self._norm_edge_axes(x, k, j)
            return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)
        rows = rows_for_axes(x, self.reduce_axes)
        y = self._norm_rows(rows)
        return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)

    def _norm_edge_axes(self, x: torch.Tensor, k: int, j: int) -> torch.Tensor:
        """Norm a prefix and a suffix of the axes without permuting the tensor.

        Two passes in the tensor's own layout: the trailing axes reduce as
        contiguous rows into fp32 partials, then the leading axes fold down the
        columns of those partials. ``l2`` takes its square root and ``inf`` its
        NaN-carrying bit-pattern max at the fold.
        """
        lead, kept, trail, planner, cfg = edge_axis_plan(
            tuple(x.shape), k, j, self._elem_bytes, self._smem_budget
        )
        dtype_str = self.dtype_to_str(self.dtype)
        if planner.needs_tiling:
            stage = _vector_norm_kernel_tiled(
                lead * kept, trail, self.op_kind, dtype_str, cfg["tile_n"], partial=True
            )
        else:
            stage = _vector_norm_kernel(lead * kept, trail, self.op_kind, dtype_str, partial=True)
        partials = stage(cfg["block_m"], cfg["threads"])(x.reshape(lead * kept, trail))
        partials = partials.reshape(lead, kept)
        if self.op_kind == "inf":
            return _inf_merge_kernel(lead, kept, dtype_str, DEFAULT_THREADS)()(partials)
        epilogue = "sqrt" if self.op_kind == "l2" else ""
        return reduce_down_rows(partials, "sum", "float32", dtype_str, 0.0, epilogue)

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
