"""Max pooling over an NCHW plane, each thread's windows read into registers.

A thread takes the outputs of one output row that a 16-byte load of an input row
covers. For each input row its windows span, it loads that run and the few elements
the windows reach past it on either side; no block stages the plane in shared memory.
``MaxPool2dRegisterKernel`` stores the maxima, ``MaxPool2dIndicesRegisterKernel`` the
maxima and their positions.
"""

import functools
from typing import ClassVar, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.pool.call_spec import (
    MaxPool2dFwdInterface,
    MaxPool2dIndicesFwdInterface,
    MaxPoolCall,
)

__all__ = ["MaxPool2dIndicesRegisterKernel", "MaxPool2dRegisterKernel"]


@functools.lru_cache(maxsize=32)
def _max_pool2d_register_kernel(
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
    with_indices,
    threads,
    evict_first,
):
    """Build the program for ``planes`` planes of ``h_in`` x ``w_in`` elements.

    A thread owns ``outputs`` outputs of one output row: a ``run`` of each input row
    its windows span, ``pad_w`` elements before it and ``after`` past it, held in
    ``window`` in float32. A row or column past the plane is read at the plane's edge
    and reports that edge's position: padding is at most half a window, so that edge
    lies in the same window, and neither the maximum nor its position changes. The
    program also returns the positions when ``with_indices``.
    """
    run = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    accum_dtype = "float32"
    index_dtype = "int64"
    outputs = run // stride_w
    out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
    after = max(kernel_w - stride_w - pad_w, 0)
    groups = out_w // outputs
    total = planes * out_h * groups

    @tilelang.jit(
        out_idx=[1, 2] if with_indices else [1],
        compile_flags=["-include", csrc_path("streaming_load.h")],
    )
    def build():
        def column(start, i):
            """The column of ``window[i]``, a column past the plane read at its edge."""
            return T.max(T.min(start - pad_w + i, w_in - 1), 0)

        @T.macro
        def pool(x, y, indices, bx):
            tx = T.get_thread_binding()
            # A thread past the last group reads the last group's runs and stores
            # nothing: the loads then issue without a branch around them.
            index = T.min(bx * threads + tx, total - 1)
            group = index % groups
            oh = (index // groups) % out_h
            plane = index // (groups * out_h)
            start = group * run
            values = T.alloc_local([run], dtype)
            window = T.alloc_local([pad_w + run + after], accum_dtype)
            peaks = T.alloc_local([outputs], accum_dtype)
            maxima = T.alloc_local([outputs], dtype)
            if with_indices:
                at = T.alloc_local([outputs], "int32")
                positions = T.alloc_local([outputs], index_dtype)
            # A window of nothing but -inf reports its first tap, as torch does.
            top = T.max(oh * stride_h - pad_h, 0)
            for e in T.unroll(outputs):
                peaks[e] = -T.infinity(accum_dtype)
                if with_indices:
                    at[e] = top * w_in + column(start, e * stride_w)
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
                    window[pad_w + i] = T.cast(values[i], accum_dtype)
                if pad_w:
                    for i in T.unroll(pad_w):
                        window[i] = T.cast(x[plane, row, column(start, i)], accum_dtype)
                if after:
                    for i in T.unroll(after):
                        window[pad_w + run + i] = T.cast(
                            x[plane, row, column(start, pad_w + run + i)], accum_dtype
                        )
                # Taps in torch's order, row by row, left to right. Strict > keeps the
                # first maximum; a NaN takes the position and holds it, so the last NaN
                # in the window wins. Written here the test lowers to FSEL; built from
                # nested T.if_then_else in a helper it did not, and ran up to 40% slower.
                for e in T.unroll(outputs):
                    for kw in T.unroll(kernel_w):
                        tap = window[e * stride_w + kw]
                        if with_indices:
                            take = T.isnan(tap) or tap > peaks[e]
                            peaks[e] = T.if_then_else(take, tap, peaks[e])
                            at[e] = T.if_then_else(
                                take, row * w_in + column(start, e * stride_w + kw), at[e]
                            )
                        else:
                            peaks[e] = T.if_then_else(T.isnan(tap) or tap > peaks[e], tap, peaks[e])
            for e in T.unroll(outputs):
                maxima[e] = T.cast(peaks[e], dtype)
                if with_indices:
                    positions[e] = T.cast(at[e], index_dtype)
            if bx * threads + tx < total:
                for e in T.vectorized(outputs):
                    y[plane, oh, group * outputs + e] = maxima[e]
                if with_indices:
                    for e in T.vectorized(outputs):
                        indices[plane, oh, group * outputs + e] = positions[e]

        if with_indices:

            @T.prim_func
            def main(
                x: T.Tensor[(planes, h_in, w_in), dtype],
                y: T.Tensor[(planes, out_h, out_w), dtype],
                indices: T.Tensor[(planes, out_h, out_w), index_dtype],
            ):
                with T.Kernel(T.ceildiv(total, threads), threads=threads) as bx:
                    pool(x, y, indices, bx)

        else:

            @T.prim_func
            def main(
                x: T.Tensor[(planes, h_in, w_in), dtype],
                y: T.Tensor[(planes, out_h, out_w), dtype],
            ):
                with T.Kernel(T.ceildiv(total, threads), threads=threads) as bx:
                    pool(x, y, None, bx)

        return main

    return build


class _MaxPool2dRegisterKernelBase(Kernel):
    """Max pooling over planes, a thread's windows read into registers.

    Serves windows without dilation or ``ceil_mode``. Along the width, the stride must
    divide the elements of a 16-byte load, so that one covers whole outputs; the outputs
    must divide into those loads, stay within the padded row, and reach at most
    one element past a load on either side. Concrete kernels set ``_with_indices`` and
    shape what ``_launch`` returns.
    """

    _with_indices: ClassVar[bool]

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: MaxPoolCall) -> bool:
        # The window-row loop is unrolled, so a window of more than 16 rows stays on the
        # one-output-a-thread kernel. A thread reads at most one element past its load on
        # either side of a row, each a load of its own.
        max_rows, max_reach = 16, 1
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32) or len(call.size) != 2:
            return False
        (h_in, w_in), (kernel_h, kernel_w) = call.size, call.window
        (stride_h, stride_w), (pad_h, pad_w) = call.stride, call.pad
        run = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        if call.ceil_mode or any(d != 1 for d in call.dilation) or run % stride_w:
            return False
        out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
        out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
        return (
            kernel_h <= max_rows
            and pad_w <= max_reach
            and kernel_w - stride_w - pad_w <= max_reach
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
        # Inputs up to 128 MiB are read evict-first in L2. A read line is then the first a
        # later allocation replaces, so fewer dirty lines are written back during the kernel:
        # at (2048, 56, 56) fp32 the maxima took 10.85 to 10.08 us and the write-back with
        # indices fell from 22.6 to 19.3 MB, while (8192, 112, 112) fp32 slowed from 121.4
        # to 125.6 us.
        evict_first = planes * h_in * w_in * dtype.itemsize <= 128 << 20
        self.kernel = _max_pool2d_register_kernel(
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
            type(self)._with_indices,
            threads=128,
            evict_first=evict_first,
        )
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def _launch(self, x: torch.Tensor):
        """Run the program on ``x``; return its leading dims and the program's outputs."""
        self._require_cuda(x=x)
        x = x.contiguous()
        # The kernel reads 16-byte vectors from the start of each row.
        x = vector_aligned(x)
        return x.shape[:-2], self.kernel()(x.view(self.planes, self.h_in, self.w_in))


class MaxPool2dRegisterKernel(_MaxPool2dRegisterKernelBase, MaxPool2dFwdInterface):
    """Max pooling over planes, a thread's windows read into registers."""

    _with_indices = False
    preferred_over = frozenset({"max_pool2d_kernel"})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, y = self._launch(x)
        return y.view(*batch, *y.shape[-2:])


class MaxPool2dIndicesRegisterKernel(_MaxPool2dRegisterKernelBase, MaxPool2dIndicesFwdInterface):
    """Max pooling with indices over planes, a thread's windows read into registers."""

    _with_indices = True
    preferred_over = frozenset({"max_pool2d_with_indices_kernel"})

    @classmethod
    def applies(cls, call: MaxPoolCall) -> bool:
        # A position is held in int32.
        return super().applies(call) and call.size[0] * call.size[1] <= 1 << 31

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, (y, indices) = self._launch(x)
        shape = (*batch, *y.shape[-2:])
        return y.view(shape), indices.view(shape)
