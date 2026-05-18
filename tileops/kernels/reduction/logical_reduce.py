"""Logical reduce kernels (any, all, count_nonzero) using TileLang.

Casts input to bool (0/1 as float32), then reduces:
  - any: reduce_max (1 if any element is non-zero)
  - all: reduce_min (1 if all elements are non-zero)
  - count_nonzero: reduce_sum (count of non-zero elements per row)

Operates on raw 2D (M, N) tensors; the kernel handles 256-element alignment
padding internally via masked loads with the appropriate identity value.

Output is bool for any/all, int64 for count_nonzero.
"""

import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    align_up,
    compute_tile_n,
    device_smem_budget,
)

__all__ = ["LogicalReduceKernel", "to_logical_float32"]

_LOGICAL_REDUCE_KINDS = {"any", "all", "count_nonzero"}

# TileLang does not support bool, integer, or complex dtypes as a storage
# dtype for T.alloc_shared / T.copy. When the caller's dtype is one of these,
# we use float32 internally instead. The Op layer is responsible for
# pre-converting the tensor to float32 before calling the kernel.
_FLOAT32_STORAGE_DTYPE = torch.float32
_UNSUPPORTED_STORAGE_DTYPES = frozenset(
    {
        torch.bool,
        torch.complex64,
        torch.complex128,
        torch.int32,
        torch.int64,
    }
)


def to_logical_float32(x: torch.Tensor) -> torch.Tensor:
    """Convert an unsupported-storage-dtype tensor to float32 for kernel dispatch.

    - bool:        True -> 1.0, False -> 0.0
    - int32/int64: direct cast to float32
    - complex:     nonzero (either real or imaginary part != 0) -> 1.0, else 0.0
    """
    if x.dtype in (torch.bool, torch.int32, torch.int64):
        return x.to(torch.float32)
    # complex: element is "truthy" if real != 0 OR imag != 0
    return ((x.real != 0) | (x.imag != 0)).to(torch.float32)


def _pad_value_for_op(op_kind: str) -> float:
    """Return the identity value for masked padding."""
    if op_kind == "all":
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Logical reduce kernel
# ---------------------------------------------------------------------------


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
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                bool_vals = T.alloc_fragment((block_m, N_padded), "float32")
                result = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment(
                    (block_m,), "int8" if op_kind != "count_nonzero" else "int64"
                )

                if _needs_pad:
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < N),
                            T.cast(x[pid_m * block_m + i, j], "float32"),
                            T.cast(_pad_val, "float32"),
                        )
                else:
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

    Iterates over the reduction dimension in chunks of ``tile_n`` columns,
    avoiding TileLang's single-fragment column limit at 32768 columns.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    _pad_val = _pad_value_for_op(op_kind)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.macro
        def compute(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int8" if op_kind != "count_nonzero" else "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_f32 = T.alloc_fragment((block_m, tile_n), "float32")
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
                    for i, j in T.Parallel(block_m, tile_n):
                        x_f32[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                            T.cast(
                                x[pid_m * block_m + i, t * tile_n + j],
                                "float32",
                            ),
                            T.cast(_pad_val, "float32"),
                        )

                    for i, j in T.Parallel(block_m, tile_n):
                        bool_vals[i, j] = T.if_then_else(x_f32[i, j] != 0.0, 1.0, 0.0)

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
    out_raw = _logical_reduce_kernel(M, N, op_kind, dtype_str)(block_m, threads)(x)
    if op_kind == "count_nonzero":
        return out_raw.to(torch.int64)
    return out_raw.bool()


@torch.library.custom_op("top::logical_reduce_tiled_fwd", mutates_args=())
def _logical_reduce_tiled_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    tile_n: int,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    out_raw = _logical_reduce_kernel_tiled(
        M, N, op_kind, dtype_str, tile_n
    )(block_m, threads)(x)
    if op_kind == "count_nonzero":
        return out_raw.to(torch.int64)
    return out_raw.bool()


@_logical_reduce_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, x):
    if op_kind == "count_nonzero":
        return torch.empty((M,), dtype=torch.int64, device=x.device)
    return torch.empty((M,), dtype=torch.bool, device=x.device)


@_logical_reduce_tiled_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, tile_n, block_m, threads, x):
    if op_kind == "count_nonzero":
        return torch.empty((M,), dtype=torch.int64, device=x.device)
    return torch.empty((M,), dtype=torch.bool, device=x.device)


# ---------------------------------------------------------------------------
# LogicalReduceKernel class
# ---------------------------------------------------------------------------


class LogicalReduceKernel(Kernel):
    """Any / all / count_nonzero forward kernel.

    Supports SM80+ architectures. Handles 256-element alignment padding inside
    the kernel. Casts input to bool (0/1) and reduces via max (any), min (all),
    or sum (count_nonzero). Uses an N-tiled fallback for long rows that exceed
    TileLang's single-fragment column limit.

    Output dtype is bool for any/all and int64 for count_nonzero.

    Note: TileLang does not support bool, integer, or complex dtypes as a
    storage dtype for shared memory. When dtype is one of these, the kernel is
    compiled for float32 internally and the Op layer is responsible for
    pre-converting the input tensor to float32 before calling forward().

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "any", "all", "count_nonzero".
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
        # TileLang cannot handle bool, integer, or complex dtypes as
        # shared-memory storage dtypes; remap all unsupported dtypes -> float32.
        self._kernel_dtype = (
            _FLOAT32_STORAGE_DTYPE if dtype in _UNSUPPORTED_STORAGE_DTYPES else dtype
        )
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch.tensor([], dtype=self._kernel_dtype).element_size()
        self._smem_budget = device_smem_budget()
        self._needs_tiling = (
            self.N_padded > MAX_SINGLE_TILE_COLS
            or self.N_padded * self._elem_bytes > self._smem_budget
        )
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
            if "tile_n" not in self.config or self.config["tile_n"] == 0:
                self.config["tile_n"] = self._tile_n_for_block_m(bm)

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed)."""
        budget = self._smem_budget
        if self.N_padded <= MAX_SINGLE_TILE_COLS:
            single = compute_tile_n(
                block_m, self._elem_bytes, self.N_padded, budget=budget,
            )
            if single == self.N_padded:
                return 0

        col_budget = MAX_SINGLE_TILE_COLS * block_m * self._elem_bytes
        effective_budget = min(budget, col_budget)
        return compute_tile_n(
            block_m, self._elem_bytes, self.N_padded, budget=effective_budget,
        )

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget."""
        if not self._needs_tiling:
            smem_per_row = self.N_padded * self._elem_bytes
            max_block_m = self._smem_budget // smem_per_row
            block_m = 1
            for bm in [1, 2, 4, 8]:
                if bm <= max_block_m:
                    block_m = bm
            return {"block_m": block_m, "threads": 128}

        best_bm = 1
        best_tile_n = self._tile_n_for_block_m(1)
        for bm in [2, 4, 8]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
            curr_num = (self.N_padded + tn - 1) // tn
            if curr_num < best_num:
                best_bm = bm
                best_tile_n = tn
        return {"block_m": best_bm, "threads": 128, "tile_n": best_tile_n}

    @property
    def autotune_configs(self) -> list[dict]:
        if not self._needs_tiling:
            smem_per_row = self.N_padded * self._elem_bytes
            max_block_m = self._smem_budget // smem_per_row
            block_ms = [bm for bm in [1, 2, 4, 8] if bm <= max_block_m]
            threads_list = [128, 256]
            configs = list(itertools.product(block_ms, threads_list))
            return [{"block_m": bm, "threads": t} for bm, t in configs]

        configs = []
        for bm in [1, 2, 4, 8]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn == 0:
                continue
            for t in [128, 256]:
                configs.append({"block_m": bm, "threads": t, "tile_n": tn})
        return configs if configs else [self.default_config]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune logical reduce, benchmarking tiled configs directly."""
        if not self._needs_tiling:
            return super().autotune(warmup=warmup, rep=rep)

        configs = self.autotune_configs
        if not configs:
            self.config = self.default_config
            return

        print(f'Start autotuning {self.__class__.__name__} (tiled path)...')

        device = torch.cuda.current_device()
        x = torch.randn(self.M, self.N, dtype=self._kernel_dtype, device=device)

        best_config = configs[0]
        best_time = float('inf')

        for cfg in configs:
            self.config = cfg
            for _ in range(warmup):
                self.forward(x)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(rep):
                self.forward(x)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / rep

            if elapsed < best_time:
                best_time = elapsed
                best_config = cfg

        self.config = best_config
        print(f'Best config: {self.config}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the any/all/count_nonzero kernel.

        Args:
            x: Input tensor of shape (M, N). Must have dtype matching
               _kernel_dtype (float32 when the original dtype is bool).

        Returns:
            Output tensor of shape (M,) with dtype bool (any/all) or int64
            (count_nonzero).
        """
        if self._needs_tiling:
            return _logical_reduce_tiled_fwd_wrapped(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_to_str(self._kernel_dtype),
                self.config["tile_n"],
                self.config["block_m"],
                self.config["threads"],
                x,
            )
        return _logical_reduce_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_to_str(self._kernel_dtype),
            self.config["block_m"],
            self.config["threads"],
            x,
        )
