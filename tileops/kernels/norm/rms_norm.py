"""RMS Norm kernel using TileLang.

y = x * rsqrt(mean(x^2) + eps) * weight

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared memory
instructions. Padding zeros don't affect sum of squares; division uses original N
for correct mean computation.
"""

import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["RmsNormKernel"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


@functools.lru_cache(maxsize=32)
def _rms_norm_kernel(M, N, eps, dtype):
    N_padded = _align_up(N, ALIGNMENT)

    @tilelang.jit(out_idx=[2])
    def _func(block_m, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                xsq_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                sumsq = T.alloc_fragment((block_m,), "float32")
                rrms = T.alloc_fragment((block_m,), "float32")

                # Load input row block
                T.copy(x[pid_m * block_m, 0], shared_buf)
                T.copy(shared_buf, x_local)

                # Compute x^2 in fp32
                for i, j in T.Parallel(block_m, N_padded):
                    xsq_f32[i, j] = (
                        T.cast(x_local[i, j], "float32") * T.cast(x_local[i, j], "float32")
                    )

                # Sum of squares along hidden dim
                T.reduce_sum(xsq_f32, sumsq, dim=1)

                # rrms = rsqrt(mean(x^2) + eps), using original N (not padded)
                for i in T.Parallel(block_m):
                    rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                # y = x * rrms * weight, result stored back in x_local
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = (
                        T.cast(x_local[i, j], "float32") * rrms[i] * T.cast(weight[j], "float32")
                    )

                # Write output
                T.copy(x_local, shared_buf)
                T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


@torch.library.custom_op("top::rms_norm_fwd", mutates_args=())
def _rms_norm_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return _rms_norm_kernel(M, N, eps, dtype_str)(block_m, threads)(x, weight)


@_rms_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, threads, x, weight):
    N_padded = _align_up(N, ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


class RmsNormKernel(Kernel):
    """RMS Norm kernel.

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
        self.kernel = _rms_norm_kernel(self.M, self.N, self.eps, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # Shared memory budget: 1 buffer * block_m * N_padded * dtype_size < 48KB
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

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _rms_norm_wrapped(
            self.M,
            self.N,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
            weight,
        )
