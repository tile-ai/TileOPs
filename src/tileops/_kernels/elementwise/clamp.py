"""Clamp kernels: scalar bounds and tensor bounds."""

import tilelang
import tilelang.language as T

from tileops._kernels.build_target import target_cache
from tileops._kernels.elementwise._base import MultiInputElementwiseKernel, ScalarParamUnaryKernel
from tileops._kernels.elementwise._nan import nan_max, nan_min
from tileops._kernels.elementwise.call_spec import (
    BoundedUnaryFwdInterface,
    BoundsCall,
    ClampTensorCall,
    ClampTensorFwdInterface,
)
from tileops._kernels.kernel_base import Entry

__all__ = [
    "ClampFwdKernel",
    "ClampTensorFwdKernel",
]


def _clamp(x, lo, hi):
    """*x* clamped to ``[lo, hi]`` in *x*'s type, NaN where any operand is NaN.

    A ``None`` bound is not applied.
    """
    if lo is not None:
        x = nan_max(x, lo)
    if hi is not None:
        x = nan_min(x, hi)
    return x


class ClampFwdKernel(ScalarParamUnaryKernel, BoundedUnaryFwdInterface):
    """Clamp: y = clamp(x, min, max); a ``None`` bound is not applied.

    Compares in the input's dtype against the bounds rounded to it, which equals torch's
    float32 comparison rounded at the store, since rounding is monotone. A NaN input or
    bound gives NaN, as in ``torch.clamp``.
    """

    @classmethod
    def entry_for(cls, call: BoundsCall) -> Entry:
        return call, lambda: cls(
            call.n_total, call.dtype, min_val=call.min_val, max_val=call.max_val
        )

    def _make_op_func(self, min_val, max_val):
        def op_func(x):
            lo = None if min_val is None else T.cast(min_val, x.dtype)
            hi = None if max_val is None else T.cast(max_val, x.dtype)
            return _clamp(x, lo, hi)

        return op_func


@target_cache(maxsize=32)
def _make_clamp_tensor_kernel(N, dtype, has_min, has_max):
    """Build the Tensor-bound clamp kernel: ``y = clamp(x, lo, hi)`` over flat operands of
    length *N*; ``has_min`` / ``has_max`` say which bounds are parameters."""
    if has_min and has_max:

        @tilelang.jit(out_idx=[3])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                lo: T.Tensor((N,), dtype),
                hi: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    lo_reg = T.alloc_fragment((block_size,), dtype)
                    hi_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    T.copy(lo[bx * block_size : (bx + 1) * block_size], lo_reg)
                    T.copy(hi[bx * block_size : (bx + 1) * block_size], hi_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        k = i * npt_arg + j
                        x_reg[k] = _clamp(x_reg[k], lo_reg[k], hi_reg[k])
                    T.copy(x_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

        return kernel

    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            bound: T.Tensor((N,), dtype),
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                bound_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                T.copy(bound[bx * block_size : (bx + 1) * block_size], bound_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    b = bound_reg[k]
                    x_reg[k] = _clamp(x_reg[k], b, None) if has_min else _clamp(x_reg[k], None, b)
                T.copy(x_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class ClampTensorFwdKernel(MultiInputElementwiseKernel, ClampTensorFwdInterface):
    """Tensor-bound clamp: ``y = clamp(x, lo, hi)``; the bound a call omits is not a
    PrimFunc parameter.

    Compares in the input's dtype, which equals torch's float32 comparison rounded at the
    store. A NaN in ``x``, ``lo`` or ``hi`` makes the output NaN, as in ``torch.clamp``.
    """

    @classmethod
    def entry_for(cls, call: ClampTensorCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype, call.has_min, call.has_max)

    def __init__(self, N_total, dtype, has_min, has_max, config=None):
        if not (has_min or has_max):
            raise ValueError("ClampTensorFwdKernel requires has_min or has_max to be True")
        self.has_min = bool(has_min)
        self.has_max = bool(has_max)
        self.INPUTS = (
            ("x", "tile"),
            *((("lo", "tile"),) if self.has_min else ()),
            *((("hi", "tile"),) if self.has_max else ()),
        )
        super().__init__(N_total, dtype, config=config)

    _builder = staticmethod(_make_clamp_tensor_kernel)

    def _builder_args(self):
        return (self.has_min, self.has_max)

    def _element(self, x, *bounds):
        bounds = list(bounds)
        lo = bounds.pop(0) if self.has_min else None
        hi = bounds.pop(0) if self.has_max else None
        return _clamp(x, lo, hi)

    def forward(self, x, lo=None, hi=None):
        return self._run(x=x, lo=lo, hi=hi)
