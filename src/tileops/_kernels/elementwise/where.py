"""The where kernel."""

import functools

import tilelang
import tilelang.language as T

from tileops._csrc import csrc_include
from tileops._kernels.elementwise._base import MultiInputElementwiseKernel
from tileops._kernels.elementwise._builders import multi_input_block
from tileops._kernels.elementwise.call_spec import WhereFwdInterface

__all__ = [
    "WhereFwdKernel",
]


def _select(cond, x, y):
    return T.if_then_else(cond != T.cast(0, "uint8"), x, y)


@functools.lru_cache(maxsize=32)
def _make_where_kernel(N, dtype, threads=256, npt=8):
    """Build where kernel: out = cond ? x : y.

    ``WhereFwdKernel.forward`` packs the bool condition as uint8 so that it
    loads as a vector (TileLang does not vectorize bool tensors). Each uint8
    element is 0 or 1, unpacked per element with a != 0 comparison.
    """

    @tilelang.jit(out_idx=[3], compile_flags=csrc_include("streaming_load.h"))
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            cond: T.Tensor((N,), "uint8"),
            x: T.Tensor((N,), dtype),
            y_in: T.Tensor((N,), dtype),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, threads_arg * npt_arg), threads=threads_arg) as bx:
                multi_input_block(bx, threads_arg, npt_arg, N, [cond, x, y_in], out, _select)

        return main

    return kernel


class WhereFwdKernel(MultiInputElementwiseKernel, WhereFwdInterface):
    """Where: out = cond ? x : y."""

    INPUTS = (("cond", "mask"), ("x", "tile"), ("y", "tile"))

    @staticmethod
    def _builder_fn():
        return _make_where_kernel

    def forward(self, cond, x, y):
        return self._run(cond=cond, x=x, y=y)
