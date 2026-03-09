"""Fused Add + Norm kernels using TileLang.

FusedAddLayerNorm: y = LayerNorm(x + residual)
FusedAddRmsNorm:   y = RmsNorm(x + residual)

Both variants return dual outputs (y, x + residual).

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared memory
instructions. Padding zeros are handled consistently with the standalone norm kernels.
"""

import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["FusedAddLayerNormKernel", "FusedAddRmsNormKernel"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Fused Add + LayerNorm kernel
# ---------------------------------------------------------------------------


def _fused_add_layer_norm_kernel(M, N, eps, dtype):
    N_padded = _align_up(N, ALIGNMENT)
    pad_count = N_padded - N

    @tilelang.jit(out_idx=[4, 5])
    def _func(block_m, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            residual: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            bias: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
            pre_norm: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_x = T.alloc_shared((block_m, N_padded), dtype)
                shared_r = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                r_local = T.alloc_fragment((block_m, N_padded), dtype)
                h_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                # Load x and residual via shared memory
                T.copy(x[pid_m * block_m, 0], shared_x)
                T.copy(shared_x, x_local)
                T.copy(residual[pid_m * block_m, 0], shared_r)
                T.copy(shared_r, r_local)

                # Fused add: h = x + residual (in fp32)
                for i, j in T.Parallel(block_m, N_padded):
                    h_f32[i, j] = T.cast(x_local[i, j], "float32") + T.cast(r_local[i, j], "float32")

                # Store pre_norm = x + residual (cast back to original dtype)
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = h_f32[i, j]
                T.copy(x_local, shared_x)
                T.copy(shared_x, pre_norm[pid_m * block_m, 0])

                # --- Mean reduction ---
                T.reduce_sum(h_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(N)

                # --- Centered variance reduction ---
                # Rewrite h_f32 in-place with (h - mean)^2
                # Save a copy of h_f32 for output computation: reuse r_local
                for i, j in T.Parallel(block_m, N_padded):
                    r_local[i, j] = h_f32[i, j]

                for i, j in T.Parallel(block_m, N_padded):
                    h_f32[i, j] = (h_f32[i, j] - mean_val[i]) * (h_f32[i, j] - mean_val[i])

                T.reduce_sum(h_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(
                        (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                        / float(N)
                        + eps
                    )

                # --- Output: y = (h - mean) * rstd * weight + bias ---
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = (
                        (T.cast(r_local[i, j], "float32") - mean_val[i])
                        * rstd[i]
                        * T.cast(weight[j], "float32")
                        + T.cast(bias[j], "float32")
                    )

                # Write y output via shared memory
                T.copy(x_local, shared_x)
                T.copy(shared_x, y[pid_m * block_m, 0])

        return main

    return _func


@torch.library.custom_op("top::fused_add_layer_norm_fwd", mutates_args=())
def _fused_add_layer_norm_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> list[torch.Tensor]:
    return _fused_add_layer_norm_kernel(M, N, eps, dtype_str)(block_m, threads)(
        x, residual, weight, bias
    )


@_fused_add_layer_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, threads, x, residual, weight, bias):
    N_padded = _align_up(N, ALIGNMENT)
    y = torch.empty((M, N_padded), dtype=x.dtype, device=x.device)
    pre_norm = torch.empty((M, N_padded), dtype=x.dtype, device=x.device)
    return [y, pre_norm]


class FusedAddLayerNormKernel(Kernel):
    """Fused Add + LayerNorm kernel.

    y = LayerNorm(x + residual), also returns x + residual.

    Supports SM80+ architectures. Uses 256-element alignment (512 bytes for
    fp16/bf16) for shared memory copies.
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
        self.kernel = _fused_add_layer_norm_kernel(self.M, self.N, self.eps, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # Shared memory budget: 2 buffers * block_m * N_padded * dtype_size < 48KB
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // (2 * smem_per_row)
        block_m = 1
        for bm in [1, 2, 4, 8, 16]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // (2 * smem_per_row)
        block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    ) -> list[torch.Tensor]:
        return _fused_add_layer_norm_wrapped(
            self.M,
            self.N,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
            residual,
            weight,
            bias,
        )


# ---------------------------------------------------------------------------
# Fused Add + RmsNorm kernel
# ---------------------------------------------------------------------------


def _fused_add_rms_norm_kernel(M, N, eps, dtype):
    N_padded = _align_up(N, ALIGNMENT)

    @tilelang.jit(out_idx=[3, 4])
    def _func(block_m, threads):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            residual: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
            pre_norm: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_x = T.alloc_shared((block_m, N_padded), dtype)
                shared_r = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                r_local = T.alloc_fragment((block_m, N_padded), dtype)
                h_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                sumsq = T.alloc_fragment((block_m,), "float32")
                rrms = T.alloc_fragment((block_m,), "float32")

                # Load x and residual via shared memory
                T.copy(x[pid_m * block_m, 0], shared_x)
                T.copy(shared_x, x_local)
                T.copy(residual[pid_m * block_m, 0], shared_r)
                T.copy(shared_r, r_local)

                # Fused add: h = x + residual (in fp32)
                for i, j in T.Parallel(block_m, N_padded):
                    h_f32[i, j] = T.cast(x_local[i, j], "float32") + T.cast(r_local[i, j], "float32")

                # Store pre_norm = x + residual (cast back to original dtype)
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = h_f32[i, j]
                T.copy(x_local, shared_x)
                T.copy(shared_x, pre_norm[pid_m * block_m, 0])

                # Compute h^2 in fp32 for sum of squares
                for i, j in T.Parallel(block_m, N_padded):
                    r_local[i, j] = h_f32[i, j] * h_f32[i, j]

                # Cast r_local to float32 for reduce_sum
                # Actually r_local is dtype, we need a float32 buffer for reduction
                # Reuse h_f32 after saving h values
                # Save h_f32 into x_local first, then compute squares in h_f32
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = h_f32[i, j]  # save h in x_local (original dtype)

                for i, j in T.Parallel(block_m, N_padded):
                    h_f32[i, j] = h_f32[i, j] * h_f32[i, j]

                # Sum of squares along hidden dim
                T.reduce_sum(h_f32, sumsq, dim=1)

                # rrms = rsqrt(mean(h^2) + eps), using original N (not padded)
                for i in T.Parallel(block_m):
                    rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                # y = h * rrms * weight
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = (
                        T.cast(x_local[i, j], "float32") * rrms[i] * T.cast(weight[j], "float32")
                    )

                # Write y output via shared memory
                T.copy(x_local, shared_x)
                T.copy(shared_x, y[pid_m * block_m, 0])

        return main

    return _func


@torch.library.custom_op("top::fused_add_rms_norm_fwd", mutates_args=())
def _fused_add_rms_norm_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> list[torch.Tensor]:
    return _fused_add_rms_norm_kernel(M, N, eps, dtype_str)(block_m, threads)(
        x, residual, weight
    )


@_fused_add_rms_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, threads, x, residual, weight):
    N_padded = _align_up(N, ALIGNMENT)
    y = torch.empty((M, N_padded), dtype=x.dtype, device=x.device)
    pre_norm = torch.empty((M, N_padded), dtype=x.dtype, device=x.device)
    return [y, pre_norm]


class FusedAddRmsNormKernel(Kernel):
    """Fused Add + RmsNorm kernel.

    y = RmsNorm(x + residual), also returns x + residual.

    Supports SM80+ architectures. Uses 256-element alignment (512 bytes for
    fp16/bf16) for shared memory copies.
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
        self.kernel = _fused_add_rms_norm_kernel(self.M, self.N, self.eps, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # Shared memory budget: 2 buffers * block_m * N_padded * dtype_size < 48KB
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // (2 * smem_per_row)
        block_m = 1
        for bm in [1, 2, 4, 8]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        smem_per_row = self.N_padded * torch.tensor([], dtype=self.dtype).element_size()
        max_block_m = (48 * 1024) // (2 * smem_per_row)
        block_ms = [bm for bm in [1, 2, 4, 8] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        return _fused_add_rms_norm_wrapped(
            self.M,
            self.N,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
            residual,
            weight,
        )
