"""Fused Add + Norm forward kernels using TileLang.

FusedAddLayerNorm: y = LayerNorm(x + residual), also outputs (x + residual)
FusedAddRmsNorm:   y = RmsNorm(x + residual),   also outputs (x + residual)

Fusing the residual add into the normalization kernel eliminates one global
memory round-trip compared to separate add + norm.  Both kernels return dual
outputs ``(y, x + residual)`` so downstream residual connections can reuse the
pre-norm sum without recomputation.

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

__all__ = ["FusedAddLayerNormKernel", "FusedAddRmsNormKernel"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Fused Add + LayerNorm kernel
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
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
            residual_out: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_x = T.alloc_shared((block_m, N_padded), dtype)
                shared_r = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                r_local = T.alloc_fragment((block_m, N_padded), dtype)
                add_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                # Load x and residual via shared memory
                T.copy(x[pid_m * block_m, 0], shared_x)
                T.copy(shared_x, x_local)
                T.copy(residual[pid_m * block_m, 0], shared_r)
                T.copy(shared_r, r_local)

                # Fused add: compute (x + residual) in fp32
                for i, j in T.Parallel(block_m, N_padded):
                    add_f32[i, j] = T.cast(x_local[i, j], "float32") + T.cast(
                        r_local[i, j], "float32"
                    )

                # Store pre-norm sum back in x_local (native dtype) for output
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = add_f32[i, j]

                # --- Mean reduction ---
                T.reduce_sum(add_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(N)

                # --- Centered variance reduction ---
                for i, j in T.Parallel(block_m, N_padded):
                    add_f32[i, j] = (add_f32[i, j] - mean_val[i]) * (
                        add_f32[i, j] - mean_val[i]
                    )

                T.reduce_sum(add_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(
                        (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                        / float(N)
                        + eps
                    )

                # --- Output y: (add - mean) * rstd * weight + bias ---
                # Re-cast from x_local (which holds the pre-norm sum in native dtype)
                for i, j in T.Parallel(block_m, N_padded):
                    r_local[i, j] = (
                        (T.cast(x_local[i, j], "float32") - mean_val[i])
                        * rstd[i]
                        * T.cast(weight[j], "float32")
                        + T.cast(bias[j], "float32")
                    )

                # Write y
                T.copy(r_local, shared_x)
                T.copy(shared_x, y[pid_m * block_m, 0])

                # Write residual_out = x + residual
                T.copy(x_local, shared_r)
                T.copy(shared_r, residual_out[pid_m * block_m, 0])

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
    return list(
        _fused_add_layer_norm_kernel(M, N, eps, dtype_str)(block_m, threads)(
            x, residual, weight, bias
        )
    )


@_fused_add_layer_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, threads, x, residual, weight, bias):
    N_padded = _align_up(N, ALIGNMENT)
    y = torch.empty((M, N_padded), dtype=x.dtype, device=x.device)
    residual_out = torch.empty((M, N_padded), dtype=x.dtype, device=x.device)
    return [y, residual_out]


class FusedAddLayerNormKernel(Kernel):
    """Fused Add + LayerNorm forward kernel.

    Computes ``y = LayerNorm(x + residual)`` and returns both ``y`` and
    ``x + residual``.  The residual add is fused into the first load pass
    to save one global memory round-trip.

    Supports SM80+ architectures.  Uses 256-element alignment for shared
    memory copies.
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
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        smem_per_row = self.N_padded * elem_bytes
        max_block_m = (48 * 1024) // (2 * smem_per_row)
        block_m = 1
        for bm in [1, 2, 4, 8, 16]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        smem_per_row = self.N_padded * elem_bytes
        max_block_m = (48 * 1024) // (2 * smem_per_row)
        block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
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

@functools.lru_cache(maxsize=32)
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
            residual_out: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_x = T.alloc_shared((block_m, N_padded), dtype)
                shared_r = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                r_local = T.alloc_fragment((block_m, N_padded), dtype)
                xsq_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                sumsq = T.alloc_fragment((block_m,), "float32")
                rrms = T.alloc_fragment((block_m,), "float32")

                # Load x and residual via shared memory
                T.copy(x[pid_m * block_m, 0], shared_x)
                T.copy(shared_x, x_local)
                T.copy(residual[pid_m * block_m, 0], shared_r)
                T.copy(shared_r, r_local)

                # Fused add: x_local <- (x + residual) in native dtype
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = T.cast(x_local[i, j], "float32") + T.cast(
                        r_local[i, j], "float32"
                    )

                # Compute (x+residual)^2 in fp32
                for i, j in T.Parallel(block_m, N_padded):
                    xsq_f32[i, j] = (
                        T.cast(x_local[i, j], "float32")
                        * T.cast(x_local[i, j], "float32")
                    )

                # Sum of squares
                T.reduce_sum(xsq_f32, sumsq, dim=1)

                # rrms = rsqrt(mean(sq) + eps)
                for i in T.Parallel(block_m):
                    rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                # y = (x+residual) * rrms * weight
                for i, j in T.Parallel(block_m, N_padded):
                    r_local[i, j] = (
                        T.cast(x_local[i, j], "float32")
                        * rrms[i]
                        * T.cast(weight[j], "float32")
                    )

                # Write y
                T.copy(r_local, shared_x)
                T.copy(shared_x, y[pid_m * block_m, 0])

                # Write residual_out = x + residual
                T.copy(x_local, shared_r)
                T.copy(shared_r, residual_out[pid_m * block_m, 0])

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
    return list(
        _fused_add_rms_norm_kernel(M, N, eps, dtype_str)(block_m, threads)(
            x, residual, weight
        )
    )


@_fused_add_rms_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, threads, x, residual, weight):
    N_padded = _align_up(N, ALIGNMENT)
    y = torch.empty((M, N_padded), dtype=x.dtype, device=x.device)
    residual_out = torch.empty((M, N_padded), dtype=x.dtype, device=x.device)
    return [y, residual_out]


class FusedAddRmsNormKernel(Kernel):
    """Fused Add + RmsNorm forward kernel.

    Computes ``y = RmsNorm(x + residual)`` and returns both ``y`` and
    ``x + residual``.  The residual add is fused into the first load pass
    to save one global memory round-trip.

    Supports SM80+ architectures.  Uses 256-element alignment for shared
    memory copies.
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
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        smem_per_row = self.N_padded * elem_bytes
        max_block_m = (48 * 1024) // (2 * smem_per_row)
        block_m = 1
        for bm in [1, 2, 4, 8]:
            if bm <= max_block_m:
                block_m = bm
        return {"block_m": block_m, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        elem_bytes = torch.tensor([], dtype=self.dtype).element_size()
        smem_per_row = self.N_padded * elem_bytes
        max_block_m = (48 * 1024) // (2 * smem_per_row)
        block_ms = [bm for bm in [1, 2, 4, 8] if bm <= max_block_m]
        threads_list = [128, 256]
        configs = list(itertools.product(block_ms, threads_list))
        return [{"block_m": bm, "threads": t} for bm, t in configs]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
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
