"""Fused Add + Norm forward kernels using TileLang.

FusedAddLayerNorm: y = LayerNorm(x + residual), also outputs (x + residual)
FusedAddRMSNorm:   y = RMSNorm(x + residual),   also outputs (x + residual)

Fusing the residual add into the normalization kernel eliminates one global
memory round-trip compared to separate add + norm.  Both kernels return dual
outputs ``(y, x + residual)`` so downstream residual connections can reuse the
pre-norm sum without recomputation.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel, require_cuda
from tileops.kernels.tiling import ALIGNMENT, align_up

from ._config import select_row_config, select_row_configs

__all__ = ["FusedAddLayerNormKernel", "FusedAddRMSNormKernel"]


# Fused Add + LayerNorm kernel


@functools.lru_cache(maxsize=32)
def _fused_add_layer_norm_kernel(M, N, eps, dtype):
    N_padded = align_up(N, ALIGNMENT)
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
                    add_f32[i, j] = (add_f32[i, j] - mean_val[i]) * (add_f32[i, j] - mean_val[i])

                T.reduce_sum(add_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt(
                        (acc[i] - float(pad_count) * mean_val[i] * mean_val[i]) / float(N) + eps
                    )

                # --- Output y: (add - mean) * rstd * weight + bias ---
                # Re-cast from x_local (which holds the pre-norm sum in native dtype)
                for i, j in T.Parallel(block_m, N_padded):
                    r_local[i, j] = (T.cast(x_local[i, j], "float32") - mean_val[i]) * rstd[
                        i
                    ] * T.cast(weight[j], "float32") + T.cast(bias[j], "float32")

                # Write y
                T.copy(r_local, shared_x)
                T.copy(shared_x, y[pid_m * block_m, 0])

                # Write residual_out = x + residual
                T.copy(x_local, shared_r)
                T.copy(shared_r, residual_out[pid_m * block_m, 0])

        return main

    return _func


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
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is resolved in ``forward``, memoized by
        ``_fused_add_layer_norm_kernel``.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.N_padded = align_up(N, ALIGNMENT)
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.N_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.N_padded, self.dtype, num_buffers=2)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run fused add + LayerNorm over the trailing ``N`` elements.

        Flattening to 2-D rows happens here, as does the alignment padding the prim_func
        requires.

        Args:
            x: Input whose trailing axis is ``N``, on a CUDA device.
            residual: Residual shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.
            bias: Affine shift holding ``N`` elements, on the same device.

        Returns:
            ``[y, residual_out]``, both shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        require_cuda(self, x=x, residual=residual, weight=weight, bias=bias)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        residual = residual.reshape(-1, self.N)
        weight = weight.reshape(self.N)
        bias = bias.reshape(self.N)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _fused_add_layer_norm_kernel(rows.shape[0], self.N, self.eps, self.dtype_str)
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        pad = self.N_padded - self.N
        if pad:
            rows = F.pad(rows, (0, pad))
            residual = F.pad(residual, (0, pad))
            weight = F.pad(weight, (0, pad))
            bias = F.pad(bias, (0, pad))
        outputs = self.kernel(self.config["block_m"], self.config["threads"])(
            rows, residual, weight, bias
        )
        if pad:
            outputs = [out[:, : self.N] for out in outputs]
        return [out.reshape(original_shape) for out in outputs]


# Fused Add + RMSNorm kernel


@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_kernel(M, N, eps, dtype):
    N_padded = align_up(N, ALIGNMENT)

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
                    xsq_f32[i, j] = T.cast(x_local[i, j], "float32") * T.cast(
                        x_local[i, j], "float32"
                    )

                # Sum of squares
                T.reduce_sum(xsq_f32, sumsq, dim=1)

                # rrms = rsqrt(mean(sq) + eps)
                for i in T.Parallel(block_m):
                    rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                # y = (x+residual) * rrms * weight
                for i, j in T.Parallel(block_m, N_padded):
                    r_local[i, j] = (
                        T.cast(x_local[i, j], "float32") * rrms[i] * T.cast(weight[j], "float32")
                    )

                # Write y
                T.copy(r_local, shared_x)
                T.copy(shared_x, y[pid_m * block_m, 0])

                # Write residual_out = x + residual
                T.copy(x_local, shared_r)
                T.copy(shared_r, residual_out[pid_m * block_m, 0])

        return main

    return _func


class FusedAddRMSNormKernel(Kernel):
    """Fused Add + RMSNorm forward kernel.

    Computes ``y = RMSNorm(x + residual)`` and returns both ``y`` and
    ``x + residual``.  The residual add is fused into the first load pass
    to save one global memory round-trip.

    Supports SM80+ architectures.  Uses 256-element alignment for shared
    memory copies.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is resolved in ``forward``, memoized by
        ``_fused_add_rms_norm_kernel``.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.N_padded = align_up(N, ALIGNMENT)
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.N_padded)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.N_padded, self.dtype, num_buffers=2)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run fused add + RMSNorm over the trailing ``N`` elements.

        Flattening to 2-D rows happens here, as does the alignment padding the prim_func
        requires.

        Args:
            x: Input whose trailing axis is ``N``, on a CUDA device.
            residual: Residual shaped like *x*, on the same device.
            weight: Affine scale holding ``N`` elements, on the same device.

        Returns:
            ``[y, residual_out]``, both shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        require_cuda(self, x=x, residual=residual, weight=weight)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        residual = residual.reshape(-1, self.N)
        weight = weight.reshape(self.N)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _fused_add_rms_norm_kernel(rows.shape[0], self.N, self.eps, self.dtype_str)
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        pad = self.N_padded - self.N
        if pad:
            rows = F.pad(rows, (0, pad))
            residual = F.pad(residual, (0, pad))
            weight = F.pad(weight, (0, pad))
        outputs = self.kernel(self.config["block_m"], self.config["threads"])(
            rows, residual, weight
        )
        if pad:
            outputs = [out[:, : self.N] for out in outputs]
        return [out.reshape(original_shape) for out in outputs]
