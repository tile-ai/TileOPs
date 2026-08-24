"""Logical reduce kernels (any, all, count_nonzero) using TileLang.

Truthiness is decided as each element is loaded, then reduced:
  - any: reduce_max (1 if any element is non-zero)
  - all: reduce_min (1 if all elements are non-zero)
  - count_nonzero: reduce_sum (count of non-zero elements per row)

Operates on raw 2D (M, N) tensors; the kernel handles 256-element alignment
padding internally via masked loads with the appropriate identity value.

A bool input is read at its own width: the prim_func declares int8, which the tensor
is reinterpreted into, and writes its int8 result back the same way. Output is bool for
any/all, int64 for count_nonzero.
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

__all__ = ["LogicalReduceKernel", "storage_dtype_for", "to_logical_storage"]

_LOGICAL_REDUCE_KINDS = {"any", "all", "count_nonzero"}

# Dtypes the prim_func cannot take as a storage dtype, mapped to one it can.
#
# bool is one byte wide and its values are the byte patterns 0 and 1, so int8
# reinterprets it for free. Widening to float32 instead would read and write four times
# the bytes, across two kernels the reduction does not need.
_FLOAT32_STORAGE_DTYPE = torch.float32
_INT8_STORAGE_DTYPE = torch.int8
_BYTE_REINTERPRETED_DTYPES = frozenset({torch.bool})
_WIDENED_STORAGE_DTYPES = frozenset(
    {
        torch.complex64,
        torch.complex128,
        torch.int32,
        torch.int64,
    }
)
_UNSUPPORTED_STORAGE_DTYPES = _BYTE_REINTERPRETED_DTYPES | _WIDENED_STORAGE_DTYPES


def storage_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """The dtype the prim_func declares for an input of *dtype*."""
    if dtype in _BYTE_REINTERPRETED_DTYPES:
        return _INT8_STORAGE_DTYPE
    if dtype in _WIDENED_STORAGE_DTYPES:
        return _FLOAT32_STORAGE_DTYPE
    return dtype


def to_logical_storage(x: torch.Tensor) -> torch.Tensor:
    """Present *x* to the kernel in a storage dtype the prim_func declares.

    - bool:        reinterpreted as int8, which copies nothing.
    - int32/int64: cast to float32.
    - complex:     nonzero (either real or imaginary part != 0) -> 1.0, else 0.0.
    """
    if x.dtype in _BYTE_REINTERPRETED_DTYPES:
        return x.view(_INT8_STORAGE_DTYPE)
    if x.dtype in (torch.int32, torch.int64):
        return x.to(torch.float32)
    # complex: element is "truthy" if real != 0 OR imag != 0
    return ((x.real != 0) | (x.imag != 0)).to(torch.float32)


def _pad_value_for_op(op_kind: str) -> float:
    """Return the identity value for masked padding."""
    if op_kind == "all":
        return 1.0
    return 0.0


# Logical reduce kernel


@functools.lru_cache(maxsize=32)
def _logical_reduce_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build a TileLang any/all/count_nonzero kernel.

    Cast input to bool (0.0 or 1.0 in float32), then:
      - any: reduce_max over the row (1.0 if any element is non-zero)
      - all: reduce_min over the row (1.0 if all elements are non-zero)
      - count_nonzero: reduce_sum over the row (count of non-zero elements)

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Original hidden dimension (last dim, before padding).
        op_kind: One of "any", "all", "count_nonzero".
        dtype: TileLang dtype string (e.g. "float16", "bfloat16", "float32").

    Returns:
        A TileLang JIT-compiled kernel factory accepting (block_m, threads).
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    _pad_val = _pad_value_for_op(op_kind)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.macro
        def compute(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int8" if op_kind != "count_nonzero" else "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                bool_vals = T.alloc_fragment((block_m, N_padded), "float32")
                result = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment(
                    (block_m,), "int8" if op_kind != "count_nonzero" else "int64"
                )

                # Truthiness is decided at the load: keeping the value in a second
                # fragment would double a tile's registers to carry a number no
                # reduction below reads.
                if _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            val = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_pad_val, "float32"),
                            )
                            bool_vals[i, j] = T.if_then_else(val != 0.0, 1.0, 0.0)
                else:
                    # Load via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            bool_vals[i, j] = T.if_then_else(
                                T.cast(shared_buf[i, j], "float32") != 0.0, 1.0, 0.0
                            )

                if op_kind == "any":
                    # any: result is 1 if max(bool_vals) == 1
                    T.reduce_max(bool_vals, result, dim=1)
                elif op_kind == "all":
                    # all: result is 1 if min(bool_vals) == 1
                    T.reduce_min(bool_vals, result, dim=1)
                else:
                    # count_nonzero: sum of bool values per row
                    T.reduce_sum(bool_vals, result, dim=1)

                if op_kind == "count_nonzero":
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i], "int64")
                else:
                    # Cast result to int8 (bool representation: 0 or 1)
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i] > 0.5, "int8")

                # Write output
                T.copy(out_local, out[pid_m * block_m])

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int8" if op_kind != "count_nonzero" else "int64"],  # noqa: F821
        ):
            compute(x, out)

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _logical_reduce_kernel_tiled(M: int, N: int, op_kind: str, dtype: str, tile_n: int):
    """Build a tiled TileLang any/all/count_nonzero kernel.

    Iterates over the reduction dimension in chunks of ``tile_n`` columns, for the
    rows a single pass cannot hold.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    _pad_val = _pad_value_for_op(op_kind)
    _past_n = num_tiles * tile_n > N

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        # Only the last tile can run past N, and only the last block past the row tail.
        needs_mask = _past_n or M % block_m != 0

        @T.macro
        def compute(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int8" if op_kind != "count_nonzero" else "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                bool_vals = T.alloc_fragment((block_m, tile_n), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                tile_acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment(
                    (block_m,), "int8" if op_kind != "count_nonzero" else "int64"
                )

                if op_kind == "all":
                    T.fill(acc, 1.0)
                else:
                    T.fill(acc, 0.0)

                for t in T.Serial(num_tiles):
                    # Truthiness is decided at the load, so a tile costs one fragment.
                    # A fully in-bounds tile arrives by T.copy; only a tile that can run
                    # past N or past the row tail is read element by element.
                    if needs_mask:
                        with T.If(t < num_tiles - 1):
                            with T.Then():
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        bool_vals[i, j] = T.if_then_else(
                                            T.cast(shared_buf[i, j], "float32") != 0.0, 1.0, 0.0
                                        )
                            with T.Else():
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        val = T.if_then_else(
                                            T.And(
                                                pid_m * block_m + i < M,
                                                t * tile_n + j < N,
                                            ),
                                            T.cast(
                                                x[pid_m * block_m + i, t * tile_n + j],
                                                "float32",
                                            ),
                                            T.cast(_pad_val, "float32"),
                                        )
                                        bool_vals[i, j] = T.if_then_else(val != 0.0, 1.0, 0.0)
                    else:
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                bool_vals[i, j] = T.if_then_else(
                                    T.cast(shared_buf[i, j], "float32") != 0.0, 1.0, 0.0
                                )

                    if op_kind == "any":
                        T.reduce_max(bool_vals, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.max(acc[i], tile_acc[i])
                    elif op_kind == "all":
                        T.reduce_min(bool_vals, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.min(acc[i], tile_acc[i])
                    else:
                        T.reduce_sum(bool_vals, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]

                if op_kind == "count_nonzero":
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i], "int64")
                else:
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i] > 0.5, "int8")

                # Write output
                T.copy(out_local, out[pid_m * block_m])

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int8" if op_kind != "count_nonzero" else "int64"],  # noqa: F821
        ):
            compute(x, out)

        return main

    return _func


class LogicalReduceKernel(Kernel):
    """Any / all / count_nonzero forward kernel.

    Supports SM80+ architectures. Handles 256-element alignment padding inside
    the kernel. Casts input to bool (0/1) and reduces via max (any), min (all),
    or sum (count_nonzero). Uses an N-tiled fallback for long rows that exceed
    TileLang's single-fragment column limit.

    Output dtype is bool for any/all and int64 for count_nonzero.

    ``forward`` takes the tensor the op declares and reduces *reduce_axes* of it; the
    permute to rows and the shape of the result are this kernel's business.

    TileLang does not support bool, integer, or complex dtypes as a shared-memory storage
    dtype. When *dtype* is one of these the kernel is compiled for float32 and ``forward``
    converts the input, so an op hands over the tensor its manifest declares and this
    restriction stays inside the implementation that has it.

    Args:
        M: Rows the reduction leaves.
        N: Elements each row reduces.
        op_kind: One of "any", "all", "count_nonzero".
        dtype: Input data type (float32, float16, bfloat16, bool, complex64,
               or complex128).
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
        if op_kind not in _LOGICAL_REDUCE_KINDS:
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of {sorted(_LOGICAL_REDUCE_KINDS)}."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.reduce_axes = tuple(reduce_axes)
        self.keepdim = keepdim
        # TileLang cannot store bool, integer or complex; each maps to a storage
        # dtype it declares, bool to the same-width int8.
        self._kernel_dtype = storage_dtype_for(dtype)
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch.tensor([], dtype=self._kernel_dtype).element_size()
        self._smem_budget = device_smem_budget(device_index)
        self._planner = BlockConfigPlanner(
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
        )
        self._needs_tiling = self._planner.needs_tiling
        self.kernel = None
        if not self._needs_tiling:
            self.kernel = _logical_reduce_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_to_str(self._kernel_dtype),
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
        """Autotune logical reduce, benchmarking tiled configs directly."""
        if not self._needs_tiling:
            return super().autotune(warmup=warmup, rep=rep)
        device = torch.cuda.current_device()
        if self._kernel_dtype.is_floating_point:
            x = torch.randn(self.M, self.N, dtype=self._kernel_dtype, device=device)
        else:
            # The int8 storage a bool input is reinterpreted into has no normal
            # distribution; the sweep only needs a mix of zero and non-zero.
            x = torch.randint(0, 2, (self.M, self.N), dtype=self._kernel_dtype, device=device)
        tune_by_forward(self, x, warmup=warmup, rep=rep, forward=self._reduce_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce *reduce_axes* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device. A dtype TileLang
                cannot store is converted here.

        Returns:
            The reduced tensor, dtype bool (any/all) or int64 (count_nonzero).

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_storage(x)
        rows = rows_for_axes(x, self.reduce_axes)
        y = self._reduce_rows(rows)
        return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)

    def _reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the trailing axis of an ``(M, N)`` buffer.

        The prim_func counts in the storage dtype; the declared output dtype is applied
        here.
        """
        dtype_str = self.dtype_to_str(self._kernel_dtype)
        if self._needs_tiling:
            program = _logical_reduce_kernel_tiled(
                self.M, self.N, self.op_kind, dtype_str, self.config["tile_n"]
            )
        else:
            program = _logical_reduce_kernel(self.M, self.N, self.op_kind, dtype_str)
        counted = program(self.config["block_m"], self.config["threads"])(x)
        if self.op_kind == "count_nonzero":
            return counted.to(torch.int64)
        # The prim_func writes 0 or 1 into int8, which is bool's own representation,
        # so the declared dtype is a reinterpretation rather than a second kernel.
        return counted.view(torch.bool)
