"""Average pooling over an NCHW plane, each thread's windows read into registers.

A thread takes the outputs of one output row that a 16-byte load of an input row
covers. For each input row its windows span, it loads that run and the few elements
the windows reach past it on either side; no block stages the plane in shared memory.
"""

import functools
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.pool.call_spec import AvgPool2dFwdInterface, AvgPoolCall

__all__ = ["AvgPool2dRegisterKernel"]

_ACCUM = "float32"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Elements a thread's windows may reach past its load on either side of a row; each is a
# load of its own.
_MAX_REACH = 1
# Window rows a thread walks.
_MAX_ROWS = 16


@functools.lru_cache(maxsize=32)
def _avg_pool2d_register_kernel(
    planes, h_in, w_in, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dtype, threads
):
    """Build the program for ``planes`` planes of ``h_in`` x ``w_in`` elements.

    A thread owns ``outputs`` outputs of one output row: a ``run`` of each input row
    its windows span, ``pad_w`` elements before it and ``after`` past it, held in
    ``window`` in float32, zero outside the plane.
    """
    run = VECTOR_ACCESS_BYTES // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    outputs = run // stride_w
    out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
    after = max(kernel_w - stride_w - pad_w, 0)
    groups = out_w // outputs
    total = planes * out_h * groups

    @tilelang.jit(out_idx=[1])
    def build():
        def element(x, plane, ih, iw):
            """``x[plane, ih, iw]`` in float32, zero outside the plane."""
            row = T.max(T.min(ih, h_in - 1), 0)
            column = T.max(T.min(iw, w_in - 1), 0)
            value = T.cast(x[plane, row, column], _ACCUM)
            zero = T.cast(0, _ACCUM)
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
                window = T.alloc_local([pad_w + run + after], _ACCUM)
                totals = T.alloc_local([outputs], _ACCUM)
                means = T.alloc_local([outputs], dtype)
                for e in T.unroll(outputs):
                    totals[e] = T.cast(0, _ACCUM)
                for kh in T.unroll(kernel_h):
                    ih = oh * stride_h - pad_h + kh
                    row = T.max(T.min(ih, h_in - 1), 0)
                    for i in T.vectorized(run):
                        values[i] = x[plane, row, start + i]
                    for i in T.unroll(run):
                        window[pad_w + i] = T.if_then_else(
                            row == ih, T.cast(values[i], _ACCUM), T.cast(0, _ACCUM)
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
                    means[e] = T.cast(totals[e] / T.cast(kernel_h * kernel_w, _ACCUM), dtype)
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
    must divide into those loads, stay within the padded row, and reach at most
    ``_MAX_REACH`` element past a load on either side.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    preferred_over = frozenset({"avg_pool2d_kernel"})

    _THREADS = 128

    @classmethod
    def applies(cls, call: AvgPoolCall) -> bool:
        if call.dtype not in _DTYPES or len(call.size) != 2:
            return False
        (h_in, w_in), (kernel_h, kernel_w) = call.size, call.window
        (stride_h, stride_w), (pad_h, pad_w) = call.stride, call.pad
        run = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        if call.ceil_mode or call.divisor_override is not None or run % stride_w:
            return False
        if (pad_h or pad_w) and not call.count_include_pad:
            return False
        out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
        out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
        return (
            kernel_h <= _MAX_ROWS
            and pad_w <= _MAX_REACH
            and kernel_w - stride_w - pad_w <= _MAX_REACH
            and out_h > 0
            and out_w > 0
            and out_w % (run // stride_w) == 0
            and out_w * stride_w <= w_in
            and w_in % run == 0
        )

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
            self._THREADS,
        )
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        y = self.kernel()(x.contiguous().view(self.planes, self.h_in, self.w_in))
        return y.view(*x.shape[:-2], *y.shape[-2:])
