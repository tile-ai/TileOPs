"""LayerNorm kernel using TileLang.

y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

256-element alignment (512 bytes for fp16/bf16) is required by T.copy() shared
memory instructions. Boundary handling for non-aligned N is performed inside
the kernel, eliminating host-side padding allocations and copies. Padding zeros
contribute 0 to the mean reduction; the centered two-pass variance computation
subtracts their exact contribution to remain numerically stable for large-offset
inputs.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._config import select_row_config, select_row_configs

__all__ = ["LayerNormKernel"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def _layer_norm_kernel(M, N, eps, dtype):
    N_padded = _align_up(N, ALIGNMENT)
    needs_pad = N_padded != N
    pad_count = N_padded - N  # number of zero-padded elements per row

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N,), dtype],
            bias: T.Tensor[(N,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                if needs_pad:
                    # Retain the original values in shared memory for the
                    # output pass while the fp32 fragment is reduced below.
                    for i, j in T.Parallel(block_m, N_padded):
                        shared_buf[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < N),
                            x[pid_m * block_m + i, j],
                            T.cast(0.0, dtype),
                        )
                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                else:
                    # Preserve the vectorized copy fast path for aligned N.
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                # --- Mean reduction ---
                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(N)

                # --- Centered variance reduction ---
                # Rewrite x_f32 in-place with (x - mean)^2.
                # Padded positions (x=0) contribute mean^2; corrected below.
                for i, j in T.Parallel(block_m, N_padded):
                    x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(
                        (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                        / float(N)
                        + eps
                    )

                # --- Output: y = (x - mean) * rstd * weight + bias ---
                if needs_pad:
                    # Store only real columns.  Returning the natural shape
                    # avoids allocating and writing an M x N_padded output
                    # merely to slice it in the Op layer.
                    for i, j in T.Parallel(block_m, N_padded):
                        if T.And(pid_m * block_m + i < M, j < N):
                            y[pid_m * block_m + i, j] = (
                                (T.cast(shared_buf[i, j], "float32") - mean_val[i])
                                * rstd[i]
                                * T.cast(weight[j], "float32")
                                + T.cast(bias[j], "float32")
                            )
                else:
                    # Re-cast from x_local (original dtype) to avoid a second
                    # fp32 buffer, then retain the vectorized copy fast path.
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = (
                            (T.cast(x_local[i, j], "float32") - mean_val[i])
                            * rstd[i]
                            * T.cast(weight[j], "float32")
                            + T.cast(bias[j], "float32")
                        )
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


@torch.library.custom_op("top::layer_norm_fwd", mutates_args=())
def _layer_norm_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _layer_norm_kernel(M, N, eps, dtype_str)(block_m, threads)(x, weight, bias)


@_layer_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, threads, x, weight, bias):
    return torch.empty((M, N), dtype=x.dtype, device=x.device)


class LayerNormKernel(Kernel):
    """LayerNorm kernel.

    Supports SM80+ architectures. Uses 256-element alignment (512 bytes for
    fp16/bf16) for shared memory copies. Single shared buffer reused for
    input load and output store.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.N_padded = _align_up(N, ALIGNMENT)
        self.kernel = _layer_norm_kernel(self.M, self.N, self.eps, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.N_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.N_padded, self.dtype)

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return _layer_norm_wrapped(
            self.M,
            self.N,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
            weight,
            bias,
        )
