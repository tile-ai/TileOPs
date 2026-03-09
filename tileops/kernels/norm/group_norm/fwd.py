"""GroupNorm forward kernel using TileLang.

y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

Input is reshaped by the Op to (M, N_row) where M = N_batch * G and
N_row = (C / G) * spatial_size. Each row represents one group in one sample.

Two paths:
  - Row-wise path (N_row_padded <= SMEM_ROW_LIMIT): loads entire row into
    shared memory for row-wise normalization, following the LayerNorm pattern
    with 256-element alignment and padding correction.
  - Chunked path (N_row_padded > SMEM_ROW_LIMIT): uses blocked reduction
    along the row to avoid shared memory overflow, similar to BatchNorm's
    non-persistent path.

Weight and bias are per-channel (length C). The Op pre-expands them to match
the row layout so the kernel receives weight/bias of shape (N_row_padded,)
per group-row.
"""

import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["GroupNormKernel"]

ALIGNMENT = 256
# Row length threshold for the row-wise (single shared buffer) path.
# Beyond this, use the chunked reduction path.
# 8192 elements * 2 bytes (fp16) = 16 KB, well within shared memory.
SMEM_ROW_LIMIT = 8192


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _find_best_block_l(L: int) -> dict:
    """Find best chunked config for given L (row length).

    Similar to BatchNorm's _find_best_block_l but adapted for GroupNorm.
    """
    for threads in [256, 128, 64, 32]:
        for k in range(512 // threads, 0, -1):
            bl = threads * k
            if bl >= L:
                continue
            if L % bl == 0:
                return {"block_l": bl, "threads": threads}
    # Fallback
    for bl in [512, 256, 128, 64, 32, 16]:
        if L % bl == 0:
            return {"block_l": bl, "threads": min(256, bl)}
    raise ValueError(
        f"L={L} is not divisible by any supported block_l. "
        "L must be divisible by at least 16."
    )


# ---------------------------------------------------------------------------
# Row-wise path: entire row fits in shared memory
# ---------------------------------------------------------------------------

def _group_norm_rowwise_kernel(M, N_row, eps, dtype):
    """Row-wise GroupNorm kernel for small N_row.

    Follows the LayerNorm pattern: load entire row into shared memory,
    compute mean/variance, normalize, apply affine transform.
    """
    N_padded = _align_up(N_row, ALIGNMENT)
    pad_count = N_padded - N_row

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(M, N_padded), dtype],
            bias: T.Tensor[(M, N_padded), dtype],
            y: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                # Load input row block via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)
                T.copy(shared_buf, x_local)

                # Cast to fp32 once
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                # Mean reduction
                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(N_row)

                # Centered variance reduction
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(
                        (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                        / float(N_row)
                        + eps
                    )

                # Output: y = (x - mean) * rstd * weight + bias
                # Weight and bias are pre-expanded by the Op to (M, N_padded)
                # Load weight via shared memory
                T.copy(weight[pid_m * block_m, 0], shared_buf)
                w_local = T.alloc_fragment((block_m, N_padded), dtype)
                T.copy(shared_buf, w_local)

                # Load bias via shared memory
                T.copy(bias[pid_m * block_m, 0], shared_buf)
                b_local = T.alloc_fragment((block_m, N_padded), dtype)
                T.copy(shared_buf, b_local)

                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = (
                        (T.cast(x_local[i, j], "float32") - mean_val[i])
                        * rstd[i]
                        * T.cast(w_local[i, j], "float32")
                        + T.cast(b_local[i, j], "float32")
                    )

                # Write output via shared memory
                T.copy(x_local, shared_buf)
                T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


# ---------------------------------------------------------------------------
# Chunked path: row too large for shared memory, use blocked reduction
# ---------------------------------------------------------------------------

def _group_norm_chunked_kernel(M, N_row, eps, dtype):
    """Chunked GroupNorm kernel for large N_row.

    Uses blocked reduction along the row dimension to avoid shared memory
    overflow. Two passes: accumulate stats, then normalize.
    """
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[3], compile_flags=["-O3", "-DENABLE_BF16"])
    def _func(block_l, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_row), dtype],
            weight: T.Tensor[(M, N_row), dtype],
            bias: T.Tensor[(M, N_row), dtype],
            y: T.Tensor[(M, N_row), dtype],
        ):
            with T.Kernel(M, threads=threads) as bm:
                # Per-element accumulators
                xsum_frag = T.alloc_fragment([1, block_l], accum_dtype)
                xsq_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(xsum_frag)
                T.clear(xsq_frag)

                # Pass 1: accumulate sum(x) and sum((x-mean_approx)^2)
                for l_tile in T.Pipelined(N_row // block_l, num_stages=0):
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x[bm, l_tile * block_l + j], accum_dtype)
                        xsum_frag[_i, j] += xval
                        xsq_frag[_i, j] += xval * xval

                # Cross-thread reduction
                sum_result = T.alloc_fragment([1], accum_dtype)
                sq_result = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(xsum_frag, sum_result, dim=1)
                T.reduce_sum(xsq_frag, sq_result, dim=1)

                # Statistics
                mean_val = sum_result[0] / T.cast(N_row, accum_dtype)
                var_val = sq_result[0] / T.cast(N_row, accum_dtype) - mean_val * mean_val
                rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(
                    var_val + T.cast(eps, accum_dtype))

                # Pass 2: normalize with affine
                for l_tile in T.Pipelined(N_row // block_l, num_stages=0):
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x[bm, l_tile * block_l + j], accum_dtype)
                        wval = T.cast(weight[bm, l_tile * block_l + j], accum_dtype)
                        bval = T.cast(bias[bm, l_tile * block_l + j], accum_dtype)
                        y[bm, l_tile * block_l + j] = T.cast(
                            wval * (xval - mean_val) * rstd_val + bval, dtype)

        return main

    return _func


# ---------------------------------------------------------------------------
# Custom op wrappers for torch.compile compatibility
# ---------------------------------------------------------------------------

@torch.library.custom_op("top::group_norm_rowwise_fwd", mutates_args=())
def _group_norm_rowwise_wrapped(
    M: int,
    N_row: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _group_norm_rowwise_kernel(M, N_row, eps, dtype_str)(block_m, threads)(x, weight, bias)


@_group_norm_rowwise_wrapped.register_fake
def _(M, N_row, eps, dtype_str, block_m, threads, x, weight, bias):
    N_padded = _align_up(N_row, ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


@torch.library.custom_op("top::group_norm_chunked_fwd", mutates_args=())
def _group_norm_chunked_wrapped(
    M: int,
    N_row: int,
    eps: float,
    dtype_str: str,
    block_l: int,
    threads: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _group_norm_chunked_kernel(M, N_row, eps, dtype_str)(block_l, threads)(x, weight, bias)


@_group_norm_chunked_wrapped.register_fake
def _(M, N_row, eps, dtype_str, block_l, threads, x, weight, bias):
    return torch.empty((M, N_row), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------

class GroupNormKernel(Kernel):
    """GroupNorm forward kernel.

    Supports SM80+ architectures. Dispatches between row-wise path (small
    N_row, uses 256-element alignment) and chunked path (large N_row, uses
    blocked reduction).

    Args:
        M: Number of group-rows (N_batch * G).
        N_row: Elements per group-row (C/G * spatial_size).
        eps: Epsilon for numerical stability.
        dtype: Input/output data type.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N_row: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.N_row = N_row
        self.eps = eps
        self.dtype = dtype
        self.N_padded = _align_up(N_row, ALIGNMENT)
        self.use_rowwise = self.N_padded <= SMEM_ROW_LIMIT

        if self.use_rowwise:
            self.kernel = _group_norm_rowwise_kernel(self.M, self.N_row, self.eps, self.dtype_str)
        else:
            self.kernel = _group_norm_chunked_kernel(self.M, self.N_row, self.eps, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.use_rowwise:
            smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
            # Single shared_buf reused for x, weight, bias loads: block_m * N_padded
            max_block_m = (48 * 1024) // smem_per_row
            block_m = 1
            for bm in [1, 2, 4, 8, 16]:
                if bm <= max_block_m:
                    block_m = bm
            return {"block_m": block_m, "threads": 256}
        else:
            return _find_best_block_l(self.N_row)

    @property
    def autotune_configs(self) -> list[dict]:
        if self.use_rowwise:
            smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
            max_block_m = (48 * 1024) // smem_per_row
            block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
            threads_list = [128, 256]
            configs = list(itertools.product(block_ms, threads_list))
            return [{"block_m": bm, "threads": t} for bm, t in configs]
        else:
            seen: set = set()
            configs = []

            def _add(cfg: dict) -> None:
                key = (cfg["block_l"], cfg["threads"])
                if key not in seen:
                    seen.add(key)
                    configs.append(cfg)

            for threads in [256, 128, 64, 32]:
                for k in range(512 // threads, 0, -1):
                    bl = threads * k
                    if bl >= self.N_row or self.N_row % bl != 0:
                        continue
                    _add({"block_l": bl, "threads": threads})

            return configs if configs else [self.default_config]

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if self.use_rowwise:
            return _group_norm_rowwise_wrapped(
                self.M,
                self.N_row,
                self.eps,
                self.dtype_str,
                self.config["block_m"],
                self.config["threads"],
                x,
                weight,
                bias,
            )
        else:
            return _group_norm_chunked_wrapped(
                self.M,
                self.N_row,
                self.eps,
                self.dtype_str,
                self.config["block_l"],
                self.config["threads"],
                x,
                weight,
                bias,
            )
