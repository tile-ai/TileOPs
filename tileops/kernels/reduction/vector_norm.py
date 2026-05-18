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

__all__ = ["VectorNormKernel"]

_VECTOR_NORM_KINDS = {"l1", "l2", "inf"}


# ---------------------------------------------------------------------------
# Vector norm kernel
# ---------------------------------------------------------------------------


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
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                transformed = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                if _needs_pad:
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < N),
                            T.cast(x[pid_m * block_m + i, j], "float32"),
                            T.cast(0.0, "float32"),
                        )
                else:
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
                    # NaN handling is done at the Op layer (InfNormFwdOp)
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
                    for i, j in T.Parallel(block_m, tile_n):
                        x_f32[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                            T.cast(
                                x[pid_m * block_m + i, t * tile_n + j],
                                "float32",
                            ),
                            T.cast(0.0, "float32"),
                        )

                    if op_kind == "l1":
                        for i, j in T.Parallel(block_m, tile_n):
                            transformed[i, j] = T.abs(x_f32[i, j])
                        T.reduce_sum(transformed, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]
                    elif op_kind == "l2":
                        for i, j in T.Parallel(block_m, tile_n):
                            transformed[i, j] = x_f32[i, j] * x_f32[i, j]
                        T.reduce_sum(transformed, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]
                    else:
                        # Note: T.reduce_max does not propagate NaN.
                        # NaN handling remains in the Op layer.
                        for i, j in T.Parallel(block_m, tile_n):
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


@torch.library.custom_op("top::vector_norm_tiled_fwd", mutates_args=())
def _vector_norm_tiled_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    tile_n: int,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _vector_norm_kernel_tiled(
        M, N, op_kind, dtype_str, tile_n
    )(block_m, threads)(x)


@_vector_norm_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


@_vector_norm_tiled_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, tile_n, block_m, threads, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# VectorNormKernel class
# ---------------------------------------------------------------------------


class VectorNormKernel(Kernel):
    """L1 / L2 / Inf norm forward kernel.

    Supports SM80+ architectures. Handles 256-element alignment padding inside
    the kernel. Computes norms via abs+sum (l1), square+sum+sqrt (l2), or
    abs+max (inf). Uses an N-tiled fallback for long rows that exceed
    TileLang's single-fragment column limit.

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
        self._elem_bytes = torch.tensor([], dtype=dtype).element_size()
        self._smem_budget = device_smem_budget()
        self._needs_tiling = (
            self.N_padded > MAX_SINGLE_TILE_COLS
            or self.N_padded * self._elem_bytes > self._smem_budget
        )
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
        """Autotune vector norm, benchmarking tiled configs directly."""
        if not self._needs_tiling:
            return super().autotune(warmup=warmup, rep=rep)

        configs = self.autotune_configs
        if not configs:
            self.config = self.default_config
            return

        print(f'Start autotuning {self.__class__.__name__} (tiled path)...')

        device = torch.cuda.current_device()
        x = torch.randn(self.M, self.N, dtype=self.dtype, device=device)

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
        """Run the l1/l2/inf norm kernel.

        Args:
            x: Input tensor of shape (M, N).

        Returns:
            Output tensor of shape (M,) with same dtype as input.
        """
        if self._needs_tiling:
            return _vector_norm_tiled_fwd_wrapped(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_to_str(self.dtype),
                self.config["tile_n"],
                self.config["block_m"],
                self.config["threads"],
                x,
            )
        return _vector_norm_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_to_str(self.dtype),
            self.config["block_m"],
            self.config["threads"],
            x,
        )
