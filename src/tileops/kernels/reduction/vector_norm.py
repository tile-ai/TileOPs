"""Vector norm kernels (l1, l2, inf) using TileLang.

Computes vector norms along the last dimension:
  - l1: sum(|x|)
  - l2: sqrt(sum(x^2))
  - inf: max(|x|), reduced over IEEE bit patterns as int32 so a row holding NaN
    norms to NaN without a second look at the input

Rows of whole 16-byte vectors fold in `ReduceFoldKernel`. The row programs here handle
256-element alignment padding internally via masked loads with zero identity values.

Output dtype matches input dtype unless one is given; l1 and l2 compute in fp32.
"""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    align_up,
    down_rows_once,
    down_rows_split,
    down_rows_splits,
)
from tileops.kernels.reduction.call_spec import NORM_KINDS, ReduceCall
from tileops.kernels.reduction.reduce import (
    ReduceKernelBase,
    RowReduceKernelBase,
)

__all__ = [
    "VectorNormEdgeKernel",
    "VectorNormKernel",
]


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
def _vector_norm_kernel(
    M: int, N: int, op_kind: str, dtype: str, out_dtype: str, partial: bool = False
):
    """Build a TileLang l1/l2/inf norm kernel.

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Original hidden dimension (last dim, before padding).
        op_kind: One of "l1", "l2", "inf".
        dtype: TileLang dtype string (e.g. "float16", "bfloat16", "float32").
        out_dtype: TileLang dtype string of the output; ``"float32"`` with *partial*.
        partial: Write partials for an outer pass — no l2 sqrt. ``inf`` partials stay
            NaN-carrying values.

    Returns:
        A TileLang JIT-compiled kernel factory accepting (block_m, threads).
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    work_dtype = _WORK_DTYPE[op_kind]

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
    M: int, N: int, op_kind: str, dtype: str, out_dtype: str, tile_n: int, partial: bool = False
):
    """Build a tiled TileLang l1/l2/inf norm kernel.

    Iterates over the reduction dimension in chunks of ``tile_n`` columns,
    avoiding TileLang's single-fragment column limit at 32768 columns.
    ``partial`` writes partials for an outer pass, as in
    ``_vector_norm_kernel``.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    work_dtype = _WORK_DTYPE[op_kind]

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


class VectorNormKernel(RowReduceKernelBase):
    """L1 / L2 / Inf norm of rows that are not whole vectors, through shared memory.

    l1 and l2 accumulate in fp32; ``inf`` reduces int32 bit patterns, which is what carries
    NaN, so a row holding a NaN norms to NaN as in ``torch.linalg.vector_norm``.
    """

    general = True

    @classmethod
    def applies(cls, call: ReduceCall) -> bool:
        return call.op_kind in NORM_KINDS and not cls.whole_vector_rows(call)

    def _untiled(self) -> object:
        return _vector_norm_kernel(self.M, self.N, self.op_kind, self.dtype_str, self.out_dtype_str)

    def _tiled(self, tile_n: int) -> object:
        return _vector_norm_kernel_tiled(
            self.M, self.N, self.op_kind, self.dtype_str, self.out_dtype_str, tile_n
        )


class VectorNormEdgeKernel(ReduceKernelBase):
    """Norm a prefix and a suffix of the axes without permuting the tensor.

    The trailing axes reduce as contiguous rows into fp32 partials, tiled where a row
    exceeds one block pass, then the leading axes fold down the columns of those
    partials. ``l2`` takes its square root at the fold, split into row slices where one
    launch would leave the grid short; ``inf`` takes its NaN-carrying bit-pattern max in
    one merge launch.
    """

    @classmethod
    def applies(cls, call: ReduceCall) -> bool:
        return call.op_kind in NORM_KINDS and cls.reduces_edge_axes(call)

    def __init__(self, call: ReduceCall):
        super().__init__(call)
        self._lead, self._kept, self._trail, planner, cfg = self.edge_plan(call)
        rows = self._lead * self._kept
        if planner.needs_tiling:
            builder = _vector_norm_kernel_tiled(
                rows,
                self._trail,
                self.op_kind,
                self.dtype_str,
                "float32",
                cfg["tile_n"],
                partial=True,
            )
        else:
            builder = _vector_norm_kernel(
                rows, self._trail, self.op_kind, self.dtype_str, "float32", partial=True
            )
        self._rows_pass = builder(cfg["block_m"], cfg["threads"])
        self._splits = down_rows_splits(self._lead, self._kept)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        lead, kept = self._lead, self._kept
        partials = self._rows_pass(x.reshape(lead * kept, self._trail)).reshape(lead, kept)
        if self.op_kind == "inf":
            return _inf_merge_kernel(lead, kept, self.out_dtype_str, DEFAULT_THREADS)()(partials)
        epilogue = "sqrt" if self.op_kind == "l2" else ""
        if self._splits == 1:
            return down_rows_once(partials, "sum", "float32", self.out_dtype_str, 0.0, epilogue)
        return down_rows_split(
            partials, "sum", "float32", self.out_dtype_str, 0.0, self._splits, epilogue
        )
