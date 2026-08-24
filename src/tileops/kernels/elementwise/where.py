"""The where kernel."""

import functools

import tilelang
import tilelang.language as T
import torch

from ._base import (
    ParametricUnaryKernel,
    _broadcast_target,
    _expand_flat,
)

__all__ = [
    "WhereFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_where_kernel(N, dtype, is_fp8=False, threads=256, npt=8):
    """Build where kernel: out = cond ? x : y.

    ``WhereFwdKernel.forward`` packs the bool condition as uint8 so that T.copy
        can perform vectorized loads (TileLang does not vectorize bool tensors).
        Each uint8 element is 0 or 1; the kernel loads it into a register
        fragment and unpacks per-element with a != 0 comparison.

        For non-fp8 dtypes, writes the result back into the x register fragment
        (in-place) to reduce register pressure and avoid a fourth data-typed
        fragment allocation.
    """
    block_size = threads * npt

    if is_fp8:

        @tilelang.jit(out_idx=[3])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(
                cond: T.Tensor((N,), "uint8"),
                x: T.Tensor((N,), dtype),
                y_in: T.Tensor((N,), dtype),
                out: T.Tensor((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            out[idx] = T.if_then_else(
                                cond[idx] != T.cast(0, "uint8"),
                                x[idx],
                                y_in[idx],
                            )

            return main
    else:

        @tilelang.jit(out_idx=[3])
        def kernel(threads_arg, npt_arg):
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


class WhereFwdKernel(ParametricUnaryKernel):
    """Where: out = cond ? x : y."""

    _DEFAULT_THREADS = 512
    _skip_fp8_output = True

    @staticmethod
    def _builder_fn():
        return _make_where_kernel

    def forward(self, cond, x, y):
        self._require_cuda(cond=cond, x=x, y=y)
        out_shape = _broadcast_target(cond, x, y)
        # A bool condition is this backend's uint8 predicate; the caller passes
        # semantic bool and never names the representation.
        if cond.dtype == torch.bool:
            cond = cond.view(torch.uint8)
        result = self._compiled_fn(
            _expand_flat(cond, out_shape),
            _expand_flat(x, out_shape),
            _expand_flat(y, out_shape),
        )
        return result.reshape(out_shape)
