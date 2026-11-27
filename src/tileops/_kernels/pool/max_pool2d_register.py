"""Max pooling over an NCHW plane, each thread's windows read into registers.

A thread takes the outputs of one output row that a 16-byte load of an input row
covers. For each input row its windows span, it loads that run and the few elements
the windows reach past it on either side; no block stages the plane in shared memory.
``MaxPool2dRegisterKernel`` stores the maxima, ``MaxPool2dIndicesRegisterKernel`` the
maxima and their positions.
"""

import math
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.pool.call_spec import (
    MaxPool2dFwdInterface,
    MaxPool2dIndicesFwdInterface,
    MaxPoolCall,
)
from tileops._kernels.pool.common import (
    ACCUM_DTYPE,
    EVICT_HINT_MAX_BYTES,
    MaxPoolKernelBase,
    max_pool_out,
    register_plane_refusal,
)

__all__ = ["MaxPool2dIndicesRegisterKernel", "MaxPool2dRegisterKernel"]


@target_cache(maxsize=32)
def _max_pool2d_register_kernel(
    call: MaxPoolCall, dtype: str, with_indices: bool, load_policy: str
):
    """Build the program for ``call``'s planes.

    A thread owns ``outputs`` outputs of one output row: a ``run`` of each input row
    its windows span, ``pad_w`` elements before it and ``after`` past it, held in
    ``window`` in float32. A row or column past the plane is read at the plane's edge
    and reports that edge's position: padding is at most half a window, so that edge
    lies in the same window, and neither the maximum nor its position changes. The
    program also returns the positions when ``with_indices``. ``load_policy`` names the
    L2 policy of the 16-byte row loads: ``evict_first``, ``evict_last`` or ``default``.
    """
    planes = call.n * call.c_in
    (h_in, w_in), (kernel_h, kernel_w) = call.size, call.window
    (stride_h, stride_w), (pad_h, pad_w) = call.stride, call.pad
    out_h, out_w = max_pool_out(call)
    run = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    index_dtype = "int64"
    outputs = run // stride_w
    after = max(kernel_w - stride_w - pad_w, 0)
    groups = out_w // outputs
    total = planes * out_h * groups
    # Positions one 16-byte store holds; a 32-byte sector holds two such halves.
    half = VECTOR_ACCESS_BYTES // getattr(torch, index_dtype).itemsize
    sectors = outputs // (2 * half)
    # Four warps a block: a block shares nothing between its threads but the shuffle of a
    # lane pair, so its width only sets how finely the grid divides.
    threads = 128

    @tilelang.jit(
        out_idx=[1, 2] if with_indices else [1],
        compile_flags=csrc_include("streaming_load.h"),
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
            window = T.alloc_local([pad_w + run + after], ACCUM_DTYPE)
            peaks = T.alloc_local([outputs], ACCUM_DTYPE)
            maxima = T.alloc_local([outputs], dtype)
            if with_indices:
                at = T.alloc_local([outputs], "int32")
            for kh in T.unroll(kernel_h):
                row = T.max(T.min(oh * stride_h - pad_h + kh, h_in - 1), 0)
                if load_policy != "default":
                    T.call_extern(
                        "handle",
                        f"tileops::load16_{load_policy}",
                        T.address_of(values[0]),
                        T.address_of(x[plane, row, start]),
                    )
                else:
                    for i in T.vectorized(run):
                        values[i] = x[plane, row, start + i]
                for i in T.unroll(run):
                    window[pad_w + i] = T.cast(values[i], ACCUM_DTYPE)
                if pad_w:
                    for i in T.unroll(pad_w):
                        window[i] = T.cast(x[plane, row, column(start, i)], ACCUM_DTYPE)
                if after:
                    for i in T.unroll(after):
                        window[pad_w + run + i] = T.cast(
                            x[plane, row, column(start, pad_w + run + i)], ACCUM_DTYPE
                        )
                # Taps in torch's order, row by row, left to right. Strict > keeps the
                # first maximum; a NaN takes the position and holds it, so the last NaN
                # in the window wins. The test is written inline so that it lowers to FSEL.
                # The first tap seeds the window, so a window of nothing but -inf reports
                # it, as torch does; the explicit unroll folds the test away.
                for e in T.unroll(outputs, explicit=True):
                    for kw in T.unroll(kernel_w, explicit=True):
                        tap = window[e * stride_w + kw]
                        if kh == 0 and kw == 0:
                            peaks[e] = tap
                            if with_indices:
                                at[e] = row * w_in + column(start, e * stride_w)
                        elif with_indices:
                            take = T.isnan(tap) or tap > peaks[e]
                            peaks[e] = T.if_then_else(take, tap, peaks[e])
                            at[e] = T.if_then_else(
                                take, row * w_in + column(start, e * stride_w + kw), at[e]
                            )
                        else:
                            peaks[e] = T.if_then_else(T.isnan(tap) or tap > peaks[e], tap, peaks[e])
            for e in T.unroll(outputs):
                maxima[e] = T.cast(peaks[e], dtype)
            if bx * threads + tx < total:
                for e in T.vectorized(outputs):
                    y[plane, oh, group * outputs + e] = maxima[e]
            if with_indices and sectors:
                # Lanes 2m and 2m + 1 own consecutive outputs. Each sector of either lane
                # is stored by both: the even lane writes the sector's low 16 bytes, the odd
                # lane its high 16 bytes, and each lane sends the other the half it writes.
                odd = tx % 2
                owner = (bx * threads + tx) // 2 * 2
                sent = T.alloc_local([half], "int32")
                got = T.alloc_local([half], "int32")
                of_even = T.alloc_local([half], index_dtype)
                of_odd = T.alloc_local([half], index_dtype)
                for j in T.unroll(sectors):
                    low = j * 2 * half
                    for w in T.unroll(half):
                        sent[w] = T.if_then_else(odd == 1, at[low + w], at[low + half + w])
                        got[w] = T.shfl_xor(sent[w], 1)
                        of_even[w] = T.cast(
                            T.if_then_else(odd == 1, got[w], at[low + w]), index_dtype
                        )
                        of_odd[w] = T.cast(
                            T.if_then_else(odd == 1, at[low + half + w], got[w]), index_dtype
                        )
                    if owner < total:
                        for e in T.vectorized(half):
                            indices[owner * outputs + low + odd * half + e] = of_even[e]
                    if owner + 1 < total:
                        for e in T.vectorized(half):
                            indices[(owner + 1) * outputs + low + odd * half + e] = of_odd[e]
            elif with_indices:
                positions = T.alloc_local([outputs], index_dtype)
                for e in T.unroll(outputs):
                    positions[e] = T.cast(at[e], index_dtype)
                if bx * threads + tx < total:
                    for e in T.vectorized(outputs):
                        indices[(bx * threads + tx) * outputs + e] = positions[e]

        if with_indices:

            @T.prim_func
            def main(
                x: T.Tensor[(planes, h_in, w_in), dtype],
                y: T.Tensor[(planes, out_h, out_w), dtype],
                indices: T.Tensor[(planes * out_h * out_w,), index_dtype],
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


class _MaxPool2dRegisterKernelBase(MaxPoolKernelBase):
    """Max pooling over planes, a thread's windows read into registers.

    Serves windows without dilation or ``ceil_mode``. Along the width, the stride must
    divide the elements of a 16-byte load, so that one covers whole outputs; the outputs
    must divide into those loads, stay within the padded row, and reach at most
    one element past a load on either side.
    """

    _with_indices: ClassVar[bool]

    @classmethod
    def refusal(cls, call: MaxPoolCall) -> str | None:
        if call.ceil_mode or any(d != 1 for d in call.dilation):
            return "does not serve ceil_mode or dilation"
        return register_plane_refusal(call) or super().refusal(call)

    def _load_policy(self, call: MaxPoolCall) -> str:
        """The L2 policy of the row loads.

        With positions, rows are read evict-last at every size. The maxima alone read
        inputs up to ``EVICT_HINT_MAX_BYTES`` evict-first, so a read line is replaced before
        a dirty one and fewer lines are written back during the kernel; larger inputs use
        the default.
        """
        input_bytes = call.n * call.c_in * math.prod(call.size) * call.dtype.itemsize
        if self._with_indices or (
            self.build_target.arch == 90
            and call.dtype == torch.bfloat16
            and call.window == call.stride == (2, 2)
            and call.pad == (0, 0)
            and 8 << 20 <= input_bytes <= 16 << 20
        ):
            return "evict_last"
        return "evict_first" if input_bytes <= EVICT_HINT_MAX_BYTES else "default"

    def _build(self, call: MaxPoolCall, dtype: str):
        return _max_pool2d_register_kernel(
            call, dtype, self._with_indices, load_policy=self._load_policy(call)
        )

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]


class MaxPool2dRegisterKernel(_MaxPool2dRegisterKernelBase, MaxPool2dFwdInterface):
    """Max pooling over planes, a thread's windows read into registers."""

    _with_indices = False
    preferred_over = frozenset({"max_pool2d_kernel"})


class MaxPool2dIndicesRegisterKernel(_MaxPool2dRegisterKernelBase, MaxPool2dIndicesFwdInterface):
    """Max pooling with indices over planes, a thread's windows read into registers."""

    _with_indices = True
    preferred_over = frozenset({"max_pool2d_with_indices_kernel"})

    @classmethod
    def refusal(cls, call: MaxPoolCall) -> str | None:
        reason = super().refusal(call)
        # A position is held in int32.
        if reason is None and call.size[0] * call.size[1] > 1 << 31:
            return f"holds a position in int32, and the plane has {call.size[0] * call.size[1]}"
        return reason
