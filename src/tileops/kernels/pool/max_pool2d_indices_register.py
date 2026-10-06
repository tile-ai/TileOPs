"""Max pooling with indices over an NCHW plane, each thread's windows read into registers.

A thread takes the outputs of one output row that a 16-byte load of an input row
covers. For each input row its windows span, it loads that run and the few elements
the windows reach past it on either side; no block stages the plane in shared memory.
"""

import functools
from typing import ClassVar, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.pool.call_spec import MaxPool2dIndicesFwdInterface, MaxPoolCall

__all__ = ["MaxPool2dIndicesRegisterKernel"]

_ACCUM = "float32"
_INDEX = "int64"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Elements a thread's windows may reach past its load on either side of a row; each is a
# load of its own.
_MAX_REACH = 1
# Window rows a thread walks; the row loop is unrolled, so taller windows stay on
# MaxPool2dWithIndicesKernel.
_MAX_ROWS = 16
# Inputs up to this size are read evict-first in L2. A read line is then the first a later
# allocation replaces, so fewer dirty lines are written back during the kernel: at
# (2048, 56, 56) fp32, 19.3 MB against 22.6 MB.
_EVICT_FIRST_BYTES = 128 << 20


@functools.lru_cache(maxsize=32)
def _max_pool2d_indices_register_kernel(
    planes,
    h_in,
    w_in,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dtype,
    threads,
    evict_first,
):
    """Build the program for ``planes`` planes of ``h_in`` x ``w_in`` elements.

    A thread owns ``outputs`` outputs of one output row: a ``run`` of each input row
    its windows span, ``pad_w`` elements before it and ``after`` past it, held in
    ``window`` in float32. A row or column past the plane is read at the plane's edge
    and reports that edge's position: padding is at most half a window, so that edge
    lies in the same window, and neither the maximum nor its position changes.
    """
    run = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    outputs = run // stride_w
    out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
    after = max(kernel_w - stride_w - pad_w, 0)
    span = pad_w + run + after
    groups = out_w // outputs
    total = planes * out_h * groups

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-include", csrc_path("streaming_load.h")])
    def build():
        @T.prim_func
        def main(
            x: T.Tensor[(planes, h_in, w_in), dtype],
            y: T.Tensor[(planes, out_h, out_w), dtype],
            indices: T.Tensor[(planes, out_h, out_w), _INDEX],
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
                window = T.alloc_local([span], _ACCUM)
                columns = T.alloc_local([span], "int32")
                peaks = T.alloc_local([outputs], _ACCUM)
                at = T.alloc_local([outputs], "int32")
                maxima = T.alloc_local([outputs], dtype)
                positions = T.alloc_local([outputs], _INDEX)
                for i in T.unroll(span):
                    columns[i] = T.max(T.min(start - pad_w + i, w_in - 1), 0)
                for e in T.unroll(outputs):
                    peaks[e] = -T.infinity(_ACCUM)
                    at[e] = 0
                for kh in T.unroll(kernel_h):
                    row = T.max(T.min(oh * stride_h - pad_h + kh, h_in - 1), 0)
                    if evict_first:
                        T.call_extern(
                            "handle",
                            "tl::tileops_load16_evict_first",
                            T.address_of(values[0]),
                            T.address_of(x[plane, row, start]),
                        )
                    else:
                        for i in T.vectorized(run):
                            values[i] = x[plane, row, start + i]
                    for i in T.unroll(run):
                        window[pad_w + i] = T.cast(values[i], _ACCUM)
                    if pad_w:
                        for i in T.unroll(pad_w):
                            window[i] = T.cast(x[plane, row, columns[i]], _ACCUM)
                    if after:
                        for i in T.unroll(after):
                            window[pad_w + run + i] = T.cast(
                                x[plane, row, columns[pad_w + run + i]], _ACCUM
                            )
                    # Taps in torch's order, row by row, left to right. Strict > keeps
                    # the first maximum; a NaN takes the position and holds it, so the
                    # last NaN in the window wins.
                    for e in T.unroll(outputs):
                        for kw in T.unroll(kernel_w):
                            tap = window[e * stride_w + kw]
                            take = T.isnan(tap) or tap > peaks[e]
                            peaks[e] = T.if_then_else(take, tap, peaks[e])
                            at[e] = T.if_then_else(
                                take, row * w_in + columns[e * stride_w + kw], at[e]
                            )
                for e in T.unroll(outputs):
                    maxima[e] = T.cast(peaks[e], dtype)
                    positions[e] = T.cast(at[e], _INDEX)
                if bx * threads + tx < total:
                    for e in T.vectorized(outputs):
                        y[plane, oh, group * outputs + e] = maxima[e]
                    for e in T.vectorized(outputs):
                        indices[plane, oh, group * outputs + e] = positions[e]

        return main

    return build


class MaxPool2dIndicesRegisterKernel(Kernel, MaxPool2dIndicesFwdInterface):
    """Max pooling with indices over planes, a thread's windows read into registers.

    Serves windows without dilation or ``ceil_mode``. Along the width, the stride must
    divide the elements of a 16-byte load, so that one covers whole outputs; the outputs
    must divide into those loads, stay within the padded row, and reach at most
    ``_MAX_REACH`` element past a load on either side.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    preferred_over = frozenset({"max_pool2d_with_indices_kernel"})

    _THREADS = 128

    @classmethod
    def applies(cls, call: MaxPoolCall) -> bool:
        if call.dtype not in _DTYPES or len(call.size) != 2:
            return False
        (h_in, w_in), (kernel_h, kernel_w) = call.size, call.window
        (stride_h, stride_w), (pad_h, pad_w) = call.stride, call.pad
        run = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        if call.ceil_mode or any(d != 1 for d in call.dilation) or run % stride_w:
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
    def entry_for(cls, call: MaxPoolCall) -> Entry:
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
        evict_first = planes * h_in * w_in * dtype.itemsize <= _EVICT_FIRST_BYTES
        self.kernel = _max_pool2d_indices_register_kernel(
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
            evict_first,
        )
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._require_cuda(x=x)
        x = x.contiguous()
        # The kernel reads 16-byte vectors from the start of each row.
        x = x.clone() if x.data_ptr() % VECTOR_ACCESS_BYTES else x
        y, indices = self.kernel()(x.view(self.planes, self.h_in, self.w_in))
        shape = (*x.shape[:-2], *y.shape[-2:])
        return y.view(shape), indices.view(shape)
