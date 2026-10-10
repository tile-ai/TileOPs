"""Average pooling over an NCL row, each thread's windows read into registers.

A thread takes the outputs one 16-byte load of the row covers and reads the few
elements their windows reach past it on either side; no block stages the row in
shared memory.
"""

import functools
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.pool.call_spec import AvgPool1dFwdInterface, AvgPoolCall
from tileops.kernels.pool.common import ACCUM_DTYPE

__all__ = ["AvgPool1dRegisterKernel"]


@functools.lru_cache(maxsize=32)
def _avg_pool1d_register_kernel(rows, l_in, kernel_l, stride_l, pad_l, dtype, threads):
    """Build the program for ``rows`` rows of ``l_in`` elements.

    A thread owns ``VECTOR_ACCESS_BYTES`` of the row, ``outputs`` outputs. The windows
    of those outputs start ``pad_l`` elements before the run and end ``after`` past it;
    ``window`` holds them in float32, zero outside the row.
    """
    run = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    outputs = run // stride_l
    out_l = (l_in + 2 * pad_l - kernel_l) // stride_l + 1
    after = max(kernel_l - stride_l - pad_l, 0)
    groups = out_l // outputs

    @tilelang.jit(out_idx=[1])
    def build():
        def element(x, row, at):
            """``x[row, at]`` in float32, zero outside the row."""
            inside = T.max(T.min(at, l_in - 1), 0)
            return T.if_then_else(
                inside == at, T.cast(x[row, inside], ACCUM_DTYPE), T.cast(0, ACCUM_DTYPE)
            )

        @T.prim_func
        def main(
            x: T.Tensor[(rows, l_in), dtype],
            y: T.Tensor[(rows, out_l), dtype],
        ):
            with T.Kernel(T.ceildiv(groups, threads), rows, threads=threads) as (bx, row):
                tx = T.get_thread_binding()
                group = bx * threads + tx
                values = T.alloc_local([run], dtype)
                window = T.alloc_local([pad_l + run + after], ACCUM_DTYPE)
                totals = T.alloc_local([outputs], ACCUM_DTYPE)
                means = T.alloc_local([outputs], dtype)
                # A thread past the last group reads the last group's run and stores
                # nothing: the loads then issue without a branch around them.
                start = T.min(group, groups - 1) * run
                for i in T.vectorized(run):
                    values[i] = x[row, start + i]
                for i in T.unroll(run):
                    window[pad_l + i] = T.cast(values[i], ACCUM_DTYPE)
                if pad_l:
                    for i in T.unroll(pad_l):
                        window[i] = element(x, row, start - pad_l + i)
                if after:
                    for i in T.unroll(after):
                        window[pad_l + run + i] = element(x, row, start + run + i)
                for e in T.unroll(outputs):
                    totals[e] = T.cast(0, ACCUM_DTYPE)
                    for t in T.unroll(kernel_l):
                        totals[e] += window[e * stride_l + t]
                    # A division, not a multiply by the reciprocal: torch divides, and the
                    # two round apart.
                    means[e] = T.cast(totals[e] / T.cast(kernel_l, ACCUM_DTYPE), dtype)
                if group < groups:
                    for e in T.vectorized(outputs):
                        y[row, group * outputs + e] = means[e]

        return main

    return build


class AvgPool1dRegisterKernel(Kernel, AvgPool1dFwdInterface):
    """Average pooling over rows, a thread's windows read into registers.

    Serves windows whose every divisor is ``kernel_size``: padding counted or absent,
    no ``ceil_mode`` and no ``divisor_override``. The stride must divide the elements of
    a 16-byte load, so that one covers whole outputs; the outputs must divide into
    those loads and the windows stay within the padded row. A thread's windows reach at
    most one element past its load on either side, each one a load of its own.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    preferred_over = frozenset({"avg_pool1d_kernel"})

    @classmethod
    def refusal(cls, call: AvgPoolCall) -> "str | None":
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return f"requires float16, bfloat16 or float32, got {call.dtype}"
        if len(call.size) != 1:
            return "serves a 1-d window"
        (l_in,), (kernel_l,), (stride_l,), (pad_l,) = call.size, call.window, call.stride, call.pad
        run = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        if call.ceil_mode or call.divisor_override is not None:
            return "does not serve ceil_mode or divisor_override"
        if run % stride_l:
            return f"requires a stride dividing the {run}-element load, got {stride_l}"
        if pad_l and not call.count_include_pad:
            return "requires count_include_pad where it pads"
        out_l = (l_in + 2 * pad_l - kernel_l) // stride_l + 1
        # Each element past the load is a load of its own; at two, AvgPool1dKernel's
        # shared-memory stage is faster.
        max_reach = 1
        if pad_l > max_reach or kernel_l - stride_l - pad_l > max_reach:
            return f"reads at most {max_reach} element past a load"
        if not (
            out_l > 0
            and out_l % (run // stride_l) == 0
            and out_l * stride_l <= l_in
            and l_in % run == 0
        ):
            return f"requires whole {run}-element loads along the row"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: AvgPoolCall) -> Entry:
        identity = (
            call.n * call.c_in,
            call.size[0],
            call.window[0],
            call.stride[0],
            call.pad[0],
            call.dtype,
        )
        return identity, lambda: cls(*identity)

    def __init__(
        self, rows: int, l_in: int, kernel_l: int, stride_l: int, pad_l: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.rows, self.l_in, self.dtype = rows, l_in, dtype
        self.kernel = _avg_pool1d_register_kernel(
            rows, l_in, kernel_l, stride_l, pad_l, self.dtype_str, threads=128
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
        x = vector_aligned(x)
        x = x.contiguous()
        y = self.kernel()(x.view(self.rows, self.l_in))
        return y.view(*x.shape[:-1], y.shape[-1])
