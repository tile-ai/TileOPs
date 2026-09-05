"""RMS Norm kernel using TileLang.

y = x * rsqrt(mean(x^2) + eps) * weight

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared memory
instructions. Padding zeros don't affect sum of squares; division uses original N
for correct mean computation.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.tiling import ALIGNMENT, align_up
from tileops.utils import get_sm_count

from ._config import select_row_config, select_row_configs

__all__ = ["RMSNormKernel"]


@functools.lru_cache(maxsize=32)
def _rms_norm_kernel(M, N, eps, dtype, partial_min_elements, sm_count):
    N_padded = align_up(N, ALIGNMENT)

    @tilelang.jit(out_idx=[2])
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
            x: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                reduce_width = threads if per_thread_partial else N_padded
                xsq_f32 = T.alloc_fragment((block_m, reduce_width), "float32")
                sumsq = T.alloc_fragment((block_m,), "float32")
                rrms = T.alloc_fragment((block_m,), "float32")

                T.copy(x[pid_m * block_m, 0], shared_buf)
                if per_thread_partial:
                    T.clear(xsq_f32)
                    for i, j in T.Parallel(block_m, threads):
                        for k in T.serial(N_padded // threads):
                            v = T.cast(shared_buf[i, k * threads + j], "float32")
                            xsq_f32[i, j] += v * v

                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    # N, not N_padded: the pad contributes zero to the sum.
                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                    T.copy(shared_buf, x_local)
                else:
                    T.copy(shared_buf, x_local)

                    for i, j in T.Parallel(block_m, N_padded):
                        xsq_f32[i, j] = T.cast(x_local[i, j], "float32") * T.cast(
                            x_local[i, j], "float32"
                        )

                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    # N, not N_padded: the pad contributes zero to the sum.
                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                # y = x * rrms * weight, result stored back in x_local
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = (
                        T.cast(x_local[i, j], "float32") * rrms[i] * T.cast(weight[j], "float32")
                    )

                T.copy(x_local, shared_buf)
                T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


class RMSNormKernel(Kernel):
    """RMS Norm kernel.

    Supports SM80+ architectures. Uses 256-element alignment (512 bytes for
    fp16/bf16) for shared memory copies. Single shared buffer reused for
    input load and output store.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    # Row elements a thread must own before the walk pays. One reduction here,
    # so one walk.
    PARTIAL_MIN_ELEMENTS_PER_THREAD = 32

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
        ``_rms_norm_kernel``.
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

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Flattening to 2-D rows and a flat weight happens here, as does the alignment
        padding the prim_func requires.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, contiguous, on the same device.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: Either input is not on a CUDA device.
        """
        if not (x.is_cuda and weight.is_cuda):
            raise ValueError(
                f"{type(self).__name__} is a CUDA kernel; got x on {x.device} and weight on "
                f"{weight.device}. Another target's backend serves other devices."
            )

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        weight = weight.reshape(self.N)
        m = rows.shape[0]

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _rms_norm_kernel(
            m,
            self.N,
            self.eps,
            self.dtype_str,
            self.PARTIAL_MIN_ELEMENTS_PER_THREAD,
            # The device the input is on, not whichever is current.
            get_sm_count(x.device.index),
        )
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        pad = self.N_padded - self.N
        if pad:
            rows = F.pad(rows, (0, pad))
            weight = F.pad(weight, (0, pad))
        y = self.kernel(self.config["block_m"], self.config["threads"])(rows, weight)
        if pad:
            y = y[:, : self.N]
        return y.reshape(original_shape)
