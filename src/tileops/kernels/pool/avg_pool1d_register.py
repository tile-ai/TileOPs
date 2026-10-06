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
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.pool.call_spec import AvgPool1dFwdInterface, AvgPoolCall

__all__ = ["AvgPool1dRegisterKernel"]

_ACCUM = "float32"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Taps a window may hold; past this the halo outgrows a thread's registers.
_MAX_TAPS = 16


@functools.lru_cache(maxsize=32)
def _avg_pool1d_register_kernel(rows, l_in, kernel_l, stride_l, pad_l, dtype, threads):
    """Build the program for ``rows`` rows of ``l_in`` elements.

    A thread owns ``VECTOR_ACCESS_BYTES`` of the row, ``outputs`` outputs. The windows
    of those outputs start ``pad_l`` elements before the run and end ``after`` past it;
    ``window`` holds them in float32, zero outside the row.
    """
    run = VECTOR_ACCESS_BYTES // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    outputs = run // stride_l
    out_l = (l_in + 2 * pad_l - kernel_l) // stride_l + 1
    after = max(kernel_l - stride_l - pad_l, 0)
    groups = out_l // outputs
    inv_k = 1.0 / kernel_l

    @tilelang.jit(out_idx=[1])
    def build():
        def element(x, row, at):
            """``x[row, at]`` in float32, zero outside the row."""
            inside = T.max(T.min(at, l_in - 1), 0)
            return T.if_then_else(inside == at, T.cast(x[row, inside], _ACCUM), T.cast(0, _ACCUM))

        @T.prim_func
        def main(
            x: T.Tensor[(rows, l_in), dtype],
            y: T.Tensor[(rows, out_l), dtype],
        ):
            with T.Kernel(T.ceildiv(groups, threads), rows, threads=threads) as (bx, row):
                tx = T.get_thread_binding()
                group = bx * threads + tx
                values = T.alloc_local([run], dtype)
                window = T.alloc_local([pad_l + run + after], _ACCUM)
                means = T.alloc_local([outputs], dtype)
                # A thread past the last group reads the last group's run and stores
                # nothing: the loads then issue without a branch around them.
                start = T.min(group, groups - 1) * run
                for i in T.vectorized(run):
                    values[i] = x[row, start + i]
                for i in T.unroll(run):
                    window[pad_l + i] = T.cast(values[i], _ACCUM)
                if pad_l:
                    for i in T.unroll(pad_l):
                        window[i] = element(x, row, start - pad_l + i)
                if after:
                    for i in T.unroll(after):
                        window[pad_l + run + i] = element(x, row, start + run + i)
                for e in T.unroll(outputs):
                    total = T.alloc_var(_ACCUM)
                    total = T.cast(0, _ACCUM)
                    for t in T.unroll(kernel_l):
                        total += window[e * stride_l + t]
                    means[e] = T.cast(total * inv_k, dtype)
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
    those loads and the windows stay within the padded row.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    preferred_over = frozenset({"avg_pool1d_kernel"})

    _THREADS = 128

    @classmethod
    def applies(cls, call: AvgPoolCall) -> bool:
        if call.dtype not in _DTYPES or len(call.size) != 1:
            return False
        (l_in,), (kernel_l,), (stride_l,), (pad_l,) = call.size, call.window, call.stride, call.pad
        run = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        if call.ceil_mode or call.divisor_override is not None or run % stride_l:
            return False
        if pad_l and not call.count_include_pad:
            return False
        out_l = (l_in + 2 * pad_l - kernel_l) // stride_l + 1
        return (
            kernel_l <= _MAX_TAPS
            and out_l > 0
            and out_l % (run // stride_l) == 0
            and out_l * stride_l <= l_in
            and l_in % run == 0
        )

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
            rows, l_in, kernel_l, stride_l, pad_l, self.dtype_str, self._THREADS
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
        y = self.kernel()(x.contiguous().view(self.rows, self.l_in))
        return y.view(*x.shape[:-1], y.shape[-1])
