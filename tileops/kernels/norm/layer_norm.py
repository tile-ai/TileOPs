"""LayerNorm kernel using TileLang.

y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared memory
instructions. Padding zeros contribute 0 to sum; the centered two-pass variance
computation subtracts the exact padding bias to keep results numerically stable
even for large-offset inputs.
"""

import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["LayerNormKernel"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def _layer_norm_kernel(M, N, eps, dtype):
    N_padded = _align_up(N, ALIGNMENT)
    pad_count = N_padded - N  # number of zero-padded elements per row

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            bias: T.Tensor[(N_padded,), dtype],
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

                # Cast to fp32 once — reused across all passes
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
                # Re-cast from x_local (original dtype) to avoid a second fp32 buffer
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = (
                        (T.cast(x_local[i, j], "float32") - mean_val[i])
                        * rstd[i]
                        * T.cast(weight[j], "float32")
                        + T.cast(bias[j], "float32")
                    )

                # Write output via shared memory
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
    N_padded = _align_up(N, ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


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
        # Shared memory budget: 1 buffer * block_m * N_padded * dtype_size < 48KB
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        # For large hidden dims, prefer smaller block_m to reduce register pressure
        # and improve occupancy. For small hidden dims, use larger block_m to amortize
        # kernel launch overhead.
        block_m = 1
        for bm in [1, 2, 4, 8, 16]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

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
