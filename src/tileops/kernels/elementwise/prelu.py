"""The PReLU kernel."""

import functools

import tilelang
import tilelang.language as T

from ._base import (
    ParametricUnaryKernel,
    _flat,
)
from ._dtype import _fp8_accum_dtype_str

__all__ = [
    "PreluFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_prelu_kernel(
    N, C, inner_size, dtype, output_dtype=None, is_fp8=False, threads=256, npt=8
):
    """Build PReLU kernel: y = x if x > 0 else weight[channel] * x.

    Weight is per-channel. Channel index follows PyTorch convention:
    for flat index ``idx``, channel = (idx // inner_size) % C, where
    ``inner_size`` is the product of all dimensions after the channel dim.

    For non-fp8 dtypes, uses register_copy strategy for input/output to
    improve memory coalescing for the main data path.
    """
    out_dtype = output_dtype or dtype

    if is_fp8:
        accum = _fp8_accum_dtype_str()

        @tilelang.jit(out_idx=[2])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                weight: T.Tensor((C,), dtype),
                y: T.Tensor((N,), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        k = i * npt_arg + j
                        idx = bx * block_size + k
                        if idx < N:
                            val = x[idx]
                            ch = (idx // inner_size) % C
                            w = weight[ch]
                            v = T.cast(val, accum)
                            wf = T.cast(w, accum)
                            zero = T.cast(0, accum)
                            y[idx] = T.if_then_else(
                                v > zero, T.Cast(out_dtype, v), T.Cast(out_dtype, wf * v)
                            )

            return main
    else:

        @tilelang.jit(out_idx=[2])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                weight: T.Tensor((C,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        k = i * npt_arg + j
                        idx = bx * block_size + k
                        val = x_reg[k]
                        ch = (idx // inner_size) % C
                        w = weight[ch]
                        zero = T.cast(0, val.dtype)
                        y_reg[k] = T.if_then_else(val > zero, val, w * val)
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

    return kernel


class PreluFwdKernel(ParametricUnaryKernel):
    """PReLU: y = x if x > 0 else weight[channel] * x."""

    def __init__(self, N_total, C, inner_size, dtype, config=None, tune=False):
        self.C = C
        self.inner_size = inner_size
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_prelu_kernel

    def _builder_positional_args(self):
        return (self.N_total, self.C, self.inner_size, self.dtype_str)

    def forward(self, x, weight):
        self._require_cuda(x=x, weight=weight)
        return self._compiled_fn(_flat(x), _flat(weight)).reshape(x.shape)
