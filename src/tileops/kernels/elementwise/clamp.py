"""Clamp kernels: scalar bounds and tensor bounds."""

import functools

import tilelang
import tilelang.language as T

from ._base import (
    MultiInputElementwiseKernel,
    ScalarParamUnaryKernel,
)

__all__ = [
    "ClampFwdKernel",
    "ClampTensorFwdKernel",
]


class ClampFwdKernel(ScalarParamUnaryKernel):
    """Clamp: y = clamp(x, min, max) with optional bounds.

    Computes in float32 and casts back at the store, so a half input keeps the
    precision of the comparison. A bound the caller omitted is not applied.
    """

    def __init__(self, N_total, dtype, min_val=None, max_val=None, config=None, tune=False):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(N_total, dtype, config=config, tune=tune)

    def _param_key(self):
        return f"min_val={self.min_val!r}|max_val={self.max_val!r}"

    def _make_op_func(self):
        min_val, max_val = self.min_val, self.max_val

        def op_func(x):
            wide = T.cast(x, "float32")
            if min_val is not None:
                wide = T.max(wide, T.cast(min_val, "float32"))
            if max_val is not None:
                wide = T.min(wide, T.cast(max_val, "float32"))
            return T.Cast(x.dtype, wide)

        return op_func


@functools.lru_cache(maxsize=32)
def _make_clamp_tensor_kernel(N, dtype, has_min, has_max, threads=256, npt=8):
    """Build the Tensor-bound clamp kernel: ``y = clamp(x, lo, hi)``.

    Inputs are flat and of length *N*; ``ClampTensorFwdKernel.forward``
    broadcasts them. ``has_min`` / ``has_max`` select the three forms the
    Tensor clamp, clamp_min and clamp_max ops take.

    The result is written back into ``x``'s fragment rather than a third
    data-typed one.
    """
    if not (has_min or has_max):
        raise ValueError("_make_clamp_tensor_kernel requires has_min or has_max to be True")

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
                        x32 = T.cast(x_reg[k], "float32")
                        lo32 = T.cast(lo_reg[k], "float32")
                        hi32 = T.cast(hi_reg[k], "float32")
                        r = T.min(T.max(x32, lo32), hi32)
                        # fmaxf / fminf return their non-NaN operand where torch
                        # returns NaN. Restored after both bounds, so one bound's
                        # NaN is not clamped away by the other.
                        r = T.if_then_else(T.isnan(hi32), hi32, r)
                        r = T.if_then_else(T.isnan(lo32), lo32, r)
                        r = T.if_then_else(T.isnan(x32), x32, r)
                        x_reg[k] = T.Cast(dtype, r)
                    T.copy(x_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

        return kernel

    take_max = bool(has_min)

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
                    x32 = T.cast(x_reg[k], "float32")
                    b32 = T.cast(bound_reg[k], "float32")
                    r = T.max(x32, b32) if take_max else T.min(x32, b32)
                    # fmaxf / fminf return their non-NaN operand; torch returns NaN.
                    r = T.if_then_else(T.isnan(b32), b32, r)
                    r = T.if_then_else(T.isnan(x32), x32, r)
                    x_reg[k] = T.Cast(dtype, r)
                T.copy(x_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class ClampTensorFwdKernel(MultiInputElementwiseKernel):
    """Tensor-bound clamp: ``y = clamp(x, lo, hi)``.

    ``has_min`` / ``has_max`` select between the three forms used by the Tensor
    clamp, clamp_min and clamp_max ops; the bound a form omits is not a
    PrimFunc parameter.

    NaN semantics match ``torch.clamp`` / ``torch.clamp_min`` /
    ``torch.clamp_max``: a NaN in ``x``, ``lo`` or ``hi`` makes the output NaN
    at that position.
    """

    DEFAULT_THREADS = 512

    def __init__(self, N_total, dtype, has_min, has_max, config=None, tune=False):
        if not (has_min or has_max):
            raise ValueError("ClampTensorFwdKernel requires has_min or has_max to be True")
        self.has_min = bool(has_min)
        self.has_max = bool(has_max)
        self.INPUTS = (
            ("x", "tile"),
            *((("lo", "tile"),) if self.has_min else ()),
            *((("hi", "tile"),) if self.has_max else ()),
        )
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_clamp_tensor_kernel

    def _builder_args(self):
        return (self.has_min, self.has_max)

    def forward(self, x, lo=None, hi=None):
        return self._run(x=x, lo=lo, hi=hi)
