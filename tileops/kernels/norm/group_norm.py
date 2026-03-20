"""GroupNorm forward kernel using TileLang.

y = (x - mean) / sqrt(var + eps) * weight + bias

where mean and var are computed over (C/G, *spatial) dimensions for each of
the G groups independently. The input (N, C, *spatial) is reshaped to
(N*G, D) where D = (C/G) * spatial_size, enabling row-wise normalization
identical to LayerNorm.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions. Padding zeros contribute 0 to sum; the centered two-pass
variance computation subtracts the exact padding bias.

Weight and bias are per-channel (C elements). After reshaping, each row of
length D = (C/G) * spatial_size has its own weight/bias slice of length D,
which is tiled from the weight/bias vectors accordingly.
"""

import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["GroupNormKernel"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def _group_norm_kernel(M, D, eps, dtype):
    """Build a row-wise normalization kernel for shape (M, D_padded).

    This is the core computation shared by GroupNorm and InstanceNorm.
    The caller is responsible for reshaping input/weight/bias into (M, D_padded).

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size (before padding).
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
    """
    D_padded = _align_up(D, ALIGNMENT)
    pad_count = D_padded - D

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, D_padded), dtype],
            weight: T.Tensor[(D_padded,), dtype],
            bias: T.Tensor[(D_padded,), dtype],
            y: T.Tensor[(M, D_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                x_local = T.alloc_fragment((block_m, D_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                # Load input row block via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)
                T.copy(shared_buf, x_local)

                # Cast to fp32 once -- reused across all passes
                for i, j in T.Parallel(block_m, D_padded):
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                # --- Mean reduction ---
                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(D)

                # --- Centered variance reduction ---
                # Rewrite x_f32 in-place with (x - mean)^2.
                # Padded positions (x=0) contribute mean^2; corrected below.
                for i, j in T.Parallel(block_m, D_padded):
                    x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(
                        (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                        / float(D)
                        + eps
                    )

                # --- Output: y = (x - mean) * rstd * weight + bias ---
                for i, j in T.Parallel(block_m, D_padded):
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


@torch.library.custom_op("top::group_norm_fwd", mutates_args=())
def _group_norm_wrapped(
    M: int,
    D: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _group_norm_kernel(M, D, eps, dtype_str)(block_m, threads)(x, weight, bias)


@_group_norm_wrapped.register_fake
def _(M, D, eps, dtype_str, block_m, threads, x, weight, bias):
    D_padded = _align_up(D, ALIGNMENT)
    return torch.empty((M, D_padded), dtype=x.dtype, device=x.device)


class GroupNormKernel(Kernel):
    """GroupNorm forward kernel.

    Normalizes each group's (C/G, *spatial) slice independently.
    Input is pre-reshaped to (M, D) where M = N*G, D = (C/G)*spatial_size.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Single shared buffer reused for input load and output store.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        D: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.D = D
        self.eps = eps
        self.dtype = dtype
        self.D_padded = _align_up(D, ALIGNMENT)
        self.kernel = _group_norm_kernel(self.M, self.D, self.eps, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # Shared memory budget: 1 buffer * block_m * D_padded * dtype_size < 48KB
        smem_per_row = self.D_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_m = 1
        for bm in [1, 2, 4, 8, 16]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.D_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // smem_per_row
        block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return _group_norm_wrapped(
            self.M,
            self.D,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
            weight,
            bias,
        )
