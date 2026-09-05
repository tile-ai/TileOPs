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
from tileops.kernels.tiling import ALIGNMENT, align_up
from tileops.utils import get_sm_count

from ._config import select_row_config, select_row_configs

__all__ = ["LayerNormKernel"]


@functools.lru_cache(maxsize=32)
def _layer_norm_kernel(M, N, eps, dtype, partial_min_elements, sm_count):
    N_padded = align_up(N, ALIGNMENT)
    needs_pad = N_padded != N
    pad_count = N_padded - N  # number of zero-padded elements per row

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):
        # A partial per thread trades the fp32 fragment's N/threads registers,
        # which cap the resident warps, for a serial walk of shared memory. Only
        # a grid that oversubscribes the device is paid back for the walk.
        per_thread_partial = (
            -(-M // block_m) > sm_count
            # A thread count that does not divide the row truncates the walk.
            and N_padded % threads == 0
            and N_padded // threads >= partial_min_elements
        )

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
                reduce_width = threads if per_thread_partial else N_padded
                x_f32 = T.alloc_fragment((block_m, reduce_width), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                mean_val = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")

                if per_thread_partial:
                    if needs_pad:
                        for i, j in T.Parallel(block_m, N_padded):
                            shared_buf[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                    else:
                        T.copy(x[pid_m * block_m, 0], shared_buf)

                    T.clear(x_f32)
                    for i, j in T.Parallel(block_m, threads):
                        for k in T.serial(N_padded // threads):
                            x_f32[i, j] += T.cast(shared_buf[i, k * threads + j], "float32")

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(N)

                    # Padded positions (x=0) contribute mean^2; corrected below.
                    T.clear(x_f32)
                    for i, j in T.Parallel(block_m, threads):
                        for k in T.serial(N_padded // threads):
                            d = T.cast(shared_buf[i, k * threads + j], "float32") - mean_val[i]
                            x_f32[i, j] += d * d

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(
                            (acc[i] - float(pad_count) * mean_val[i] * mean_val[i]) / float(N) + eps
                        )

                    if not needs_pad:
                        T.copy(shared_buf, x_local)
                else:
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

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(N)

                    # Padded positions (x=0) contribute mean^2; corrected below.
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(
                            (acc[i] - float(pad_count) * mean_val[i] * mean_val[i]) / float(N) + eps
                        )

                # --- Output: y = (x - mean) * rstd * weight + bias ---
                if needs_pad:
                    # Store only real columns.  Returning the natural shape
                    # avoids allocating and writing an M x N_padded output
                    # merely to slice it in the Op layer.
                    for i, j in T.Parallel(block_m, N_padded):
                        if T.And(pid_m * block_m + i < M, j < N):
                            y[pid_m * block_m + i, j] = (
                                T.cast(shared_buf[i, j], "float32") - mean_val[i]
                            ) * rstd[i] * T.cast(weight[j], "float32") + T.cast(bias[j], "float32")
                else:
                    # Re-cast from x_local (original dtype) to avoid a second
                    # fp32 buffer, then retain the vectorized copy fast path.
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = (T.cast(x_local[i, j], "float32") - mean_val[i]) * rstd[
                            i
                        ] * T.cast(weight[j], "float32") + T.cast(bias[j], "float32")
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


class LayerNormKernel(Kernel):
    """LayerNorm kernel.

    Supports SM80+ architectures. Uses 256-element alignment (512 bytes for
    fp16/bf16) for shared memory copies. Single shared buffer reused for
    input load and output store.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    # Row elements a thread must own before the walk pays. Two reductions here,
    # so two walks, so twice the row RMSNorm needs.
    PARTIAL_MIN_ELEMENTS_PER_THREAD = 64

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
        ``_layer_norm_kernel``.
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
        return select_row_configs(self.N_padded, self.dtype)

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Flattening to 2-D rows and a flat weight and bias happens here; the prim_func
        handles the non-aligned tail itself.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, contiguous, on the same device.
            bias: Affine shift holding ``N`` elements, contiguous, on the same device.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_cuda(x=x, weight=weight, bias=bias)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        weight = weight.reshape(self.N)
        bias = bias.reshape(self.N)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _layer_norm_kernel(
            rows.shape[0],
            self.N,
            self.eps,
            self.dtype_str,
            self.PARTIAL_MIN_ELEMENTS_PER_THREAD,
            # The device the input is on, not whichever is current.
            get_sm_count(rows.device.index),
        )
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        y = self.kernel(self.config["block_m"], self.config["threads"])(rows, weight, bias)
        return y.reshape(original_shape)
