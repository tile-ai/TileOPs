"""Average pooling over an NCHW plane, each thread's windows read into registers.

A thread takes the outputs of one output row that a 16-byte load of an input row
covers. For each input row its windows span, it loads that run and the few elements
the windows reach past it on either side; no block stages the plane in shared memory.
"""

from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.pool.call_spec import AvgPool2dFwdInterface, AvgPoolCall
from tileops._kernels.pool.common import ACCUM_DTYPE, register_plane_refusal

__all__ = ["AvgPool2dRegisterKernel"]


@target_cache(maxsize=32)
def _avg_pool2d_register_kernel(
    planes, h_in, w_in, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dtype
):
    """Build the program for ``planes`` planes of ``h_in`` x ``w_in`` elements.

    A thread owns ``outputs`` outputs of one output row: a ``run`` of each input row
    its windows span, ``pad_w`` elements before it and ``after`` past it, held in
    ``window`` in float32, zero outside the plane.
    """
    run = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    outputs = run // stride_w
    out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
    after = max(kernel_w - stride_w - pad_w, 0)
    groups = out_w // outputs
    total = planes * out_h * groups
    # Four warps a block: a block shares nothing between its threads, so its width only
    # sets how finely the grid divides.
    threads = 128

    @tilelang.jit(out_idx=[1])
    def build():
        def element(x, plane, ih, iw):
            """``x[plane, ih, iw]`` in float32, zero outside the plane."""
            row = T.max(T.min(ih, h_in - 1), 0)
            column = T.max(T.min(iw, w_in - 1), 0)
            value = T.cast(x[plane, row, column], ACCUM_DTYPE)
            zero = T.cast(0, ACCUM_DTYPE)
            return T.if_then_else(row == ih, T.if_then_else(column == iw, value, zero), zero)

        @T.prim_func
        def main(
            x: T.Tensor[(planes, h_in, w_in), dtype],
            y: T.Tensor[(planes, out_h, out_w), dtype],
        ):
            with T.Kernel(T.ceildiv(total, threads), threads=threads) as bx:
                tx = T.get_thread_binding()
                # A thread past the last group reads the last group's runs and stores
                # nothing: the loads then issue without a branch around them.
                index = T.min(bx * threads + tx, total - 1)
                group = index % groups
                oh = (index // groups) % out_h
                plane = index // (groups * out_h)
                start = group * run
                values = T.alloc_local([run], dtype)
                window = T.alloc_local([pad_w + run + after], ACCUM_DTYPE)
                totals = T.alloc_local([outputs], ACCUM_DTYPE)
                means = T.alloc_local([outputs], dtype)
                for e in T.unroll(outputs):
                    totals[e] = T.cast(0, ACCUM_DTYPE)
                for kh in T.unroll(kernel_h):
                    ih = oh * stride_h - pad_h + kh
                    row = T.max(T.min(ih, h_in - 1), 0)
                    for i in T.vectorized(run):
                        values[i] = x[plane, row, start + i]
                    for i in T.unroll(run):
                        window[pad_w + i] = T.if_then_else(
                            row == ih, T.cast(values[i], ACCUM_DTYPE), T.cast(0, ACCUM_DTYPE)
                        )
                    if pad_w:
                        for i in T.unroll(pad_w):
                            window[i] = element(x, plane, ih, start - pad_w + i)
                    if after:
                        for i in T.unroll(after):
                            window[pad_w + run + i] = element(x, plane, ih, start + run + i)
                    for e in T.unroll(outputs):
                        for kw in T.unroll(kernel_w):
                            totals[e] += window[e * stride_w + kw]
                for e in T.unroll(outputs):
                    # A division, not a multiply by the reciprocal: torch divides, and the
                    # two round apart in a 16-bit result.
                    means[e] = T.cast(totals[e] / T.cast(kernel_h * kernel_w, ACCUM_DTYPE), dtype)
                if bx * threads + tx < total:
                    for e in T.vectorized(outputs):
                        y[plane, oh, group * outputs + e] = means[e]

        return main

    return build


class AvgPool2dRegisterKernel(Kernel, AvgPool2dFwdInterface):
    """Average pooling over planes, a thread's windows read into registers.

    Serves windows whose every divisor is ``kernel_h * kernel_w``: padding counted or
    absent, no ``ceil_mode`` and no ``divisor_override``. Along the width, the stride must
    divide the elements of a 16-byte load, so that one covers whole outputs; the outputs
    must divide into those loads, stay within the padded row, and reach at most one
    element past a load on either side. Windows are at most 16 rows tall.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    preferred_over = frozenset({"avg_pool2d_kernel"})

    @classmethod
    def refusal(cls, call: AvgPoolCall) -> str | None:
        if call.ceil_mode or call.divisor_override is not None:
            return "does not serve ceil_mode or divisor_override"
        if any(call.pad) and not call.count_include_pad:
            return "requires count_include_pad where it pads"
        return register_plane_refusal(call) or super().refusal(call)

    @classmethod
    def entry_for(cls, call: AvgPoolCall) -> Entry:
        identity = (
            call.n * call.c_in,
            *call.size,
            *call.window,
            *call.stride,
            *call.pad,
            call.dtype,
        )
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        planes: int,
        h_in: int,
        w_in: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.planes, self.h_in, self.w_in, self.dtype = planes, h_in, w_in, dtype
        self.kernel = _avg_pool2d_register_kernel(
            planes,
            h_in,
            w_in,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            self.dtype_str,
        )
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def _compile(self, cfg: dict):
        """The program ``forward`` launches; its builder takes no configuration."""
        return self.kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_devices(x=x)
        x = x.contiguous()
        y = self.program(self._compile)(x.view(self.planes, self.h_in, self.w_in))
        return y.view(*x.shape[:-2], *y.shape[-2:])
