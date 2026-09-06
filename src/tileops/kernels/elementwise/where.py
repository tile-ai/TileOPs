"""The where kernel."""

import functools

import tilelang
import tilelang.language as T

from ._base import MultiInputElementwiseKernel

__all__ = [
    "WhereFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_where_kernel(N, dtype, threads=256, npt=8):
    """Build where kernel: out = cond ? x : y.

    ``WhereFwdKernel.forward`` packs the bool condition as uint8 so that T.copy
    can perform vectorized loads (TileLang does not vectorize bool tensors).
    Each uint8 element is 0 or 1; the kernel loads it into a register fragment
    and unpacks per-element with a != 0 comparison.

    The result is written back into ``x``'s register fragment rather than a
    fourth data-typed one.
    """

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            cond: T.Tensor((N,), "uint8"),
            x: T.Tensor((N,), dtype),
            y_in: T.Tensor((N,), dtype),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                c_reg = T.alloc_fragment((block_size,), "uint8")
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(cond[bx * block_size : (bx + 1) * block_size], c_reg)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                T.copy(y_in[bx * block_size : (bx + 1) * block_size], y_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    x_reg[k] = T.if_then_else(
                        c_reg[k] != T.cast(0, "uint8"),
                        x_reg[k],
                        y_reg[k],
                    )
                T.copy(x_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class WhereFwdKernel(MultiInputElementwiseKernel):
    """Where: out = cond ? x : y."""

    DEFAULT_THREADS = 512
    INPUTS = (("cond", "mask"), ("x", "tile"), ("y", "tile"))

    @staticmethod
    def _builder_fn():
        return _make_where_kernel

    def forward(self, cond, x, y):
        return self._run(cond=cond, x=x, y=y)
