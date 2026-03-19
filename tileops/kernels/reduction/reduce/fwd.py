"""Reduce kernels (sum, mean, amin, amax, prod, std, var, var_mean) using TileLang.

Two kernel families:
  - _simple_reduce_kernel: single-pass reduce for sum/mean/amin/amax/prod.
  - _welford_reduce_kernel: two-pass Welford for std/var/var_mean.

Both operate on 2D (M, N_padded) tensors; the Op layer handles reshape.
256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.
"""

import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up

__all__ = ["ReduceKernel"]

# Supported simple op kinds and their T.reduce_* mapping
_SIMPLE_REDUCE_MAP = {
    "sum": "reduce_sum",
    "mean": "reduce_sum",
    "amax": "reduce_max",
    "amin": "reduce_min",
    "prod": "reduce_sum",  # prod uses log-sum-exp trick
}

_WELFORD_KINDS = {"std", "var", "var_mean"}


# ---------------------------------------------------------------------------
# Simple reduce kernel
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _simple_reduce_kernel(M, N, op_kind, dtype):
    """Build a simple reduce kernel for sum/mean/amax/amin/prod.

    For prod, we compute exp(sum(log(abs(x)))) * sign, which is numerically
    more stable than direct T.reduce_prod for large N.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)

    if op_kind == "prod":
        return _prod_reduce_kernel(M, N, dtype)

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
                acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                # Load via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)

                # Cast to fp32
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                # Reduce
                if op_kind == "sum":
                    T.reduce_sum(x_f32, acc, dim=1)
                elif op_kind == "mean":
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i] = acc[i] / float(N)
                elif op_kind == "amax":
                    T.reduce_max(x_f32, acc, dim=1)
                elif op_kind == "amin":
                    # Negate, reduce_max, negate back
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = -x_f32[i, j]
                    T.reduce_max(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i] = -acc[i]

                # Cast back to output dtype
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(acc[i], dtype)

                # Write output
                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _prod_reduce_kernel(M, N, dtype):
    """Product reduce via log-sum-exp: exp(sum(log(|x|))) * sign."""
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
                log_abs = T.alloc_fragment((block_m, N_padded), "float32")
                sign_neg = T.alloc_fragment((block_m, N_padded), "float32")
                acc_log = T.alloc_fragment((block_m,), "float32")
                acc_sign = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                T.copy(x[pid_m * block_m, 0], shared_buf)

                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                # Compute log(|x|) and count negatives
                # Padded positions have x=0; set log_abs to 0 (log(1)=0, so
                # they contribute a factor of 1 to the product).
                for i, j in T.Parallel(block_m, N_padded):
                    abs_val = T.abs(x_f32[i, j])
                    # Use a small epsilon to avoid log(0)
                    log_abs[i, j] = T.log(T.max(abs_val, 1e-38))
                    # 1 if negative, 0 if non-negative
                    sign_neg[i, j] = T.if_then_else(x_f32[i, j] < 0.0, 1.0, 0.0)

                # Zero out padded positions for log_abs (they are 0 from input
                # padding but log(0) is -inf, so override them)
                # Padding positions have x=0 after F.pad in Op, so abs_val=0,
                # log_abs=-inf. We need them to be 0 (neutral for sum).
                # But since padded x = 0, the product is 0 anyway for non-pad-safe ops.
                # The Op layer sets padded elements to 1.0 for prod.

                T.reduce_sum(log_abs, acc_log, dim=1)
                T.reduce_sum(sign_neg, acc_sign, dim=1)

                for i in T.Parallel(block_m):
                    # Subtract padding contribution (padding elements set to 1.0,
                    # so log(1.0)=0 -- no correction needed if Op pads with 1.0)
                    prod_val = T.exp(acc_log[i])
                    # If odd number of negatives, negate
                    # Use fmod to check parity
                    neg_count_mod2 = acc_sign[i] - T.floor(acc_sign[i] / 2.0) * 2.0
                    prod_val = T.if_then_else(neg_count_mod2 > 0.5, -prod_val, prod_val)
                    out_local[i] = T.cast(prod_val, dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Welford reduce kernel (std, var, var_mean)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _welford_reduce_kernel(M, N, op_kind, correction, dtype):
    """Build a Welford-based reduce kernel for std/var/var_mean."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)

    out_idx = [1, 2] if op_kind == "var_mean" else [1]

    @tilelang.jit(out_idx=out_idx)
    def _func(block_m, threads):
        if op_kind == "var_mean":

            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                out_var: T.Tensor[(M,), dtype],
                out_mean: T.Tensor[(M,), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
                    var_sum = T.alloc_fragment((block_m,), "float32")
                    out_v = T.alloc_fragment((block_m,), dtype)
                    out_m = T.alloc_fragment((block_m,), dtype)

                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    # Mean
                    T.reduce_sum(x_f32, row_sum, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = row_sum[i] / float(N)

                    # Variance: sum((x - mean)^2) / (N - correction)
                    for i, j in T.Parallel(block_m, N_padded):
                        sq_diff[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                    T.reduce_sum(sq_diff, var_sum, dim=1)

                    # Correct for padding: padded elements contribute mean^2 each
                    pad_count = N_padded - N
                    for i in T.Parallel(block_m):
                        corrected_sum = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                        variance = corrected_sum / float(N - correction)
                        out_v[i] = T.cast(variance, dtype)
                        out_m[i] = T.cast(mean_val[i], dtype)

                    T.copy(out_v, out_var[pid_m * block_m])
                    T.copy(out_m, out_mean[pid_m * block_m])

        else:
            # std or var (single output)
            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                out: T.Tensor[(M,), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
                    var_sum = T.alloc_fragment((block_m,), "float32")
                    out_local = T.alloc_fragment((block_m,), dtype)

                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    # Mean
                    T.reduce_sum(x_f32, row_sum, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = row_sum[i] / float(N)

                    # Variance
                    for i, j in T.Parallel(block_m, N_padded):
                        sq_diff[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                    T.reduce_sum(sq_diff, var_sum, dim=1)

                    pad_count = N_padded - N
                    if op_kind == "var":
                        for i in T.Parallel(block_m):
                            corrected_sum = (
                                var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            )
                            out_local[i] = T.cast(corrected_sum / float(N - correction), dtype)
                    else:  # std
                        for i in T.Parallel(block_m):
                            corrected_sum = (
                                var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            )
                            out_local[i] = T.cast(
                                T.sqrt(corrected_sum / float(N - correction)), dtype
                            )

                    T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# custom_op wrappers
# ---------------------------------------------------------------------------


@torch.library.custom_op("top::reduce_simple_fwd", mutates_args=())
def _reduce_simple_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _simple_reduce_kernel(M, N, op_kind, dtype_str)(block_m, threads)(x)


@_reduce_simple_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


@torch.library.custom_op("top::reduce_welford_fwd", mutates_args=())
def _reduce_welford_wrapped(
    M: int,
    N: int,
    op_kind: str,
    correction: int,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    kernel = _welford_reduce_kernel(M, N, op_kind, correction, dtype_str)
    result = kernel(block_m, threads)(x)
    if op_kind == "var_mean":
        # Returns (var, mean)
        return [result[0], result[1]]
    else:
        return [result]


@_reduce_welford_wrapped.register_fake
def _(M, N, op_kind, correction, dtype_str, block_m, threads, x):
    if op_kind == "var_mean":
        return [
            torch.empty((M,), dtype=x.dtype, device=x.device),
            torch.empty((M,), dtype=x.dtype, device=x.device),
        ]
    return [torch.empty((M,), dtype=x.dtype, device=x.device)]


# ---------------------------------------------------------------------------
# ReduceKernel class
# ---------------------------------------------------------------------------


class ReduceKernel(Kernel):
    """Unified reduce kernel supporting sum/mean/amin/amax/prod/std/var/var_mean.

    Supports SM80+ architectures. Uses 256-element alignment for shared memory
    copies. Dispatches to simple or Welford kernel based on op_kind.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        correction: int = 1,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.correction = correction
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._is_welford = op_kind in _WELFORD_KINDS

        if self._is_welford:
            self.kernel = _welford_reduce_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.correction,
                self.dtype_str,
            )
        else:
            self.kernel = _simple_reduce_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_str,
            )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_m = 1
        for bm in [1, 2, 4, 8]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_ms = [bm for bm in [1, 2, 4, 8] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

    def forward(self, x: torch.Tensor) -> object:
        if self._is_welford:
            results = _reduce_welford_wrapped(
                self.M,
                self.N,
                self.op_kind,
                self.correction,
                self.dtype_str,
                self.config["block_m"],
                self.config["threads"],
                x,
            )
            if self.op_kind == "var_mean":
                return results[0], results[1]
            return results[0]
        else:
            return _reduce_simple_wrapped(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_str,
                self.config["block_m"],
                self.config["threads"],
                x,
            )
