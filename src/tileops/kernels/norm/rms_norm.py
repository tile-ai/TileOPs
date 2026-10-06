"""RMSNorm kernel using TileLang.

y = x * rsqrt(mean(x^2) + eps) * weight

The normalized row goes from the register fragment straight to global memory. Only the
partial reduction reads shared memory, because a thread walking a strided run of the row
can reach it there and cannot reach another thread's registers.

The fragment may be wider than the row: columns past the row load zero and are never
stored, and the mean divides by N.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import STATIC_SHARED_BYTES
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm._config import select_row_config, select_row_configs
from tileops.kernels.norm.call_spec import LayerNormCall, RMSNormFwdInterface
from tileops.kernels.tiling import ALIGNMENT, align_up
from tileops.utils import get_sm_count

__all__ = ["RMSNormKernel"]


@functools.lru_cache(maxsize=32)
def _rms_norm_kernel(M, N, N_padded, eps, dtype, has_weight, partial_min_elements, sm_count):
    col_guard = N_padded != N

    @tilelang.jit(out_idx=[2])
    def _func(block_m, threads):
        # A partial per thread trades the fp32 fragment's N/threads registers,
        # which cap the resident warps, for a serial walk of shared memory. Only
        # a grid that oversubscribes the device is paid back for the walk.
        # A tail row block runs past the end unless every index is guarded.
        row_guard = M % block_m != 0
        per_thread_partial = (
            not col_guard
            and -(-M // block_m) > sm_count
            # A thread count that does not divide the row truncates the walk.
            and N_padded % threads == 0
            and N_padded // threads >= partial_min_elements
        )

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N if has_weight else 1,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                reduce_width = threads if per_thread_partial else N_padded
                xsq_f32 = T.alloc_fragment((block_m, reduce_width), "float32")
                sumsq = T.alloc_fragment((block_m,), "float32")
                rrms = T.alloc_fragment((block_m,), "float32")

                if per_thread_partial:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    T.copy(x[pid_m * block_m, 0], shared_buf)
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
                    for i, j in T.Parallel(block_m, N_padded):
                        if row_guard and col_guard:
                            x_local[i, j] = T.if_then_else(
                                (pid_m * block_m + i < M) & (j < N),
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                        elif row_guard:
                            x_local[i, j] = T.if_then_else(
                                pid_m * block_m + i < M,
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                        elif col_guard:
                            x_local[i, j] = T.if_then_else(
                                j < N, x[pid_m * block_m + i, j], T.cast(0.0, dtype)
                            )
                        else:
                            x_local[i, j] = x[pid_m * block_m + i, j]

                    for i, j in T.Parallel(block_m, N_padded):
                        xsq_f32[i, j] = T.cast(x_local[i, j], "float32") * T.cast(
                            x_local[i, j], "float32"
                        )

                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    # N, not N_padded: the pad contributes zero to the sum.
                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                # y = x * rrms * weight, written from the fragment holding the row.
                for i, j in T.Parallel(block_m, N_padded):
                    # Nested: `and` between a Python bool and a TIR comparison is not TIR.
                    if (not row_guard) or pid_m * block_m + i < M:  # noqa: SIM102
                        if (not col_guard) or j < N:
                            if has_weight:
                                y[pid_m * block_m + i, j] = T.cast(
                                    T.cast(x_local[i, j], "float32")
                                    * rrms[i]
                                    * T.cast(weight[j], "float32"),
                                    dtype,
                                )
                            else:
                                y[pid_m * block_m + i, j] = T.cast(
                                    T.cast(x_local[i, j], "float32") * rrms[i], dtype
                                )

        return main

    return _func


class RMSNormKernel(Kernel, RMSNormFwdInterface):
    """RMSNorm kernel.

    Supports SM80+ architectures. The row is held in a register fragment from the load
    through the store; rows of at most ``_SHARED_ROW_MAX`` elements share a block.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    # Row elements a thread must own before the walk pays. One reduction here,
    # so one walk.
    PARTIAL_MIN_ELEMENTS_PER_THREAD = 32
    # Rows at most this wide share a block of the default 128 threads: one of them gives
    # each thread less than one 16-byte access of a 16-bit dtype.
    _SHARED_ROW_MAX = 512
    # Elements such a block then holds: two 16-byte accesses per thread.
    _SHARED_BLOCK_ELEMENTS = 2048
    # Threads a row past the static shared budget gets untuned: never measurably slower
    # than 128 there, and faster as the row widens.
    _WIDE_ROW_THREADS = 256

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
        # A row sharing a block is held at a power of two, so whole rows tile the fragment.
        self.N_padded = (
            1 << (N - 1).bit_length() if N <= self._SHARED_ROW_MAX else align_up(N, ALIGNMENT)
        )
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        config = select_row_config()
        if self.N_padded <= self._SHARED_ROW_MAX:
            config["block_m"] = self._SHARED_BLOCK_ELEMENTS // self.N_padded
        elif self.N_padded * self.dtype.itemsize > STATIC_SHARED_BYTES:
            config["threads"] = self._WIDE_ROW_THREADS
        return config

    @property
    def autotune_configs(self) -> list[dict]:
        """The width-derived default alone for shared-block rows; L2-warm tuning misranks them."""
        if self.default_config["block_m"] > 1:
            return [self.default_config]
        # Past the shared budget the best width still varies with the row.
        return select_row_configs(self.N_padded, self.dtype, every_width_past_budget=True)

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, contiguous, on the same device,
                or ``None`` to scale by one.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: Either input is not on a CUDA device.
        """
        if not (x.is_cuda and (weight is None or weight.is_cuda)):
            weight_device = None if weight is None else weight.device
            raise ValueError(
                f"{type(self).__name__} is a CUDA kernel; got x on {x.device} and weight on "
                f"{weight_device}. Another target's backend serves other devices."
            )

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        has_weight = weight is not None
        # The prim_func keeps its weight parameter without one; it reads nothing from it.
        weight = weight.reshape(self.N) if has_weight else rows.new_empty(1)
        m = rows.shape[0]

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _rms_norm_kernel(
            m,
            self.N,
            self.N_padded,
            self.eps,
            self.dtype_str,
            has_weight,
            self.PARTIAL_MIN_ELEMENTS_PER_THREAD,
            # The device the input is on, not whichever is current.
            get_sm_count(x.device.index),
        )
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        block_m = self.config["block_m"]
        if self.N_padded <= self._SHARED_ROW_MAX:
            block_m = min(block_m, 1 << (m - 1).bit_length())
        y = self.kernel(block_m, self.config["threads"])(rows, weight)
        return y.reshape(original_shape)
