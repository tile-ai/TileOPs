"""Clamp kernels: scalar bounds and tensor bounds."""

import functools

import tilelang
import tilelang.language as T

from ._base import (
    ParametricUnaryKernel,
    _broadcast_target,
    _expand_flat,
)

__all__ = [
    "ClampFwdKernel",
    "ClampTensorFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_clamp_kernel(
    N,
    dtype,
    has_min,
    has_max,
    min_val,
    max_val,
    output_dtype=None,
    is_fp8=False,
    threads=256,
    npt=8,
):
    """Build clamp kernel: y = clamp(x, min_val, max_val) with optional bounds.

    For non-fp8 dtypes, uses register_copy strategy: fragment load -> compute
    -> fragment store for coalesced memory access.  Computes in fp32 then
    casts back to preserve precision.
    """
    out_dtype = output_dtype or dtype

    if is_fp8:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            v32 = T.cast(x[idx], "float32")
                            if has_min:
                                lo = T.cast(min_val, "float32")
                                v32 = T.max(v32, lo)
                            if has_max:
                                hi = T.cast(max_val, "float32")
                                v32 = T.min(v32, hi)
                            y[idx] = T.Cast(out_dtype, v32)

            return main
    else:

        @tilelang.jit(out_idx=[1])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        val = x_reg[i * npt_arg + j]
                        v32 = T.cast(val, "float32")
                        if has_min:
                            lo = T.cast(min_val, "float32")
                            v32 = T.max(v32, lo)
                        if has_max:
                            hi = T.cast(max_val, "float32")
                            v32 = T.min(v32, hi)
                        y_reg[i * npt_arg + j] = T.Cast(val.dtype, v32)
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

    return kernel


class ClampFwdKernel(ParametricUnaryKernel):
    """Clamp: y = clamp(x, min, max) with optional bounds."""

    def __init__(self, N_total, dtype, min_val=None, max_val=None, config=None, tune=False):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_clamp_kernel

    def _builder_args(self):
        return (
            self.min_val is not None,
            self.max_val is not None,
            self.min_val if self.min_val is not None else 0.0,
            self.max_val if self.max_val is not None else 0.0,
        )


@functools.lru_cache(maxsize=32)
def _make_clamp_tensor_kernel(
    N, dtype, has_min, has_max, output_dtype=None, is_fp8=False, threads=256, npt=8
):
    """Build Tensor-bound clamp kernel.

    Inputs (all flat, length N, broadcast and flattened by
        ``ClampTensorFwdKernel.forward``):
            x: data tensor.
            lo: lower-bound tensor (only present when ``has_min``).
            hi: upper-bound tensor (only present when ``has_max``).

        Output:
            y: clamp result, same dtype as ``output_dtype`` (or ``dtype``).

        For fp8 the cast/compute uses fp32 to preserve precision; for non-fp8
        the kernel uses register_copy with fp32 accumulation.

        NaN semantics: matches ``torch.clamp`` / ``torch.clamp_min`` /
        ``torch.clamp_max``. If ``x``, ``lo``, or ``hi`` is NaN at a position,
        the output at that position is NaN. ``T.max`` / ``T.min`` on CUDA do
        not propagate NaN by themselves (they return the non-NaN operand), so
        we add explicit ``isnan`` guards in fp32 -- mirroring the pattern used
        by ``MaximumFwdKernel`` / ``MinimumFwdKernel``.
    """
    if not (has_min or has_max):
        raise ValueError(
            "_make_clamp_tensor_kernel requires has_min or has_max to be True",
        )
    out_dtype = output_dtype or dtype

    if is_fp8:
        if has_min and has_max:

            @tilelang.jit(out_idx=[3])
            def kernel(threads_arg, npt_arg):
                block_size = threads_arg * npt_arg

                @T.prim_func
                def main(
                    x: T.Tensor((N,), dtype),
                    lo: T.Tensor((N,), dtype),
                    hi: T.Tensor((N,), dtype),
                    y: T.Tensor((N,), out_dtype),
                ):
                    with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                        for i, j in T.Parallel(threads_arg, npt_arg):
                            idx = (bx * threads_arg + i) * npt_arg + j
                            if idx < N:
                                x32 = T.cast(x[idx], "float32")
                                lo32 = T.cast(lo[idx], "float32")
                                hi32 = T.cast(hi[idx], "float32")
                                r = T.max(x32, lo32)
                                r = T.min(r, hi32)
                                # NaN propagation (PyTorch semantics):
                                # if any of x/lo/hi is NaN -> output NaN.
                                r = T.if_then_else(T.isnan(hi32), hi32, r)
                                r = T.if_then_else(T.isnan(lo32), lo32, r)
                                r = T.if_then_else(T.isnan(x32), x32, r)
                                y[idx] = T.Cast(out_dtype, r)

                return main

            return kernel
        if has_min:

            @tilelang.jit(out_idx=[2])
            def kernel(threads_arg, npt_arg):
                block_size = threads_arg * npt_arg

                @T.prim_func
                def main(
                    x: T.Tensor((N,), dtype),
                    lo: T.Tensor((N,), dtype),
                    y: T.Tensor((N,), out_dtype),
                ):
                    with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                        for i, j in T.Parallel(threads_arg, npt_arg):
                            idx = (bx * threads_arg + i) * npt_arg + j
                            if idx < N:
                                x32 = T.cast(x[idx], "float32")
                                lo32 = T.cast(lo[idx], "float32")
                                r = T.max(x32, lo32)
                                # NaN propagation (PyTorch clamp_min):
                                # if x or lo is NaN -> output NaN.
                                r = T.if_then_else(T.isnan(lo32), lo32, r)
                                r = T.if_then_else(T.isnan(x32), x32, r)
                                y[idx] = T.Cast(out_dtype, r)

                return main

            return kernel

        # has_max only
        @tilelang.jit(out_idx=[2])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                hi: T.Tensor((N,), dtype),
                y: T.Tensor((N,), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            x32 = T.cast(x[idx], "float32")
                            hi32 = T.cast(hi[idx], "float32")
                            r = T.min(x32, hi32)
                            # NaN propagation (PyTorch clamp_max):
                            # if x or hi is NaN -> output NaN.
                            r = T.if_then_else(T.isnan(hi32), hi32, r)
                            r = T.if_then_else(T.isnan(x32), x32, r)
                            y[idx] = T.Cast(out_dtype, r)

            return main

        return kernel

    # non-fp8 path (register_copy)
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
                        r = T.max(x32, lo32)
                        r = T.min(r, hi32)
                        # NaN propagation (PyTorch clamp):
                        # if any of x/lo/hi is NaN -> output NaN.
                        r = T.if_then_else(T.isnan(hi32), hi32, r)
                        r = T.if_then_else(T.isnan(lo32), lo32, r)
                        r = T.if_then_else(T.isnan(x32), x32, r)
                        x_reg[k] = T.Cast(dtype, r)
                    T.copy(x_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

        return kernel
    if has_min:

        @tilelang.jit(out_idx=[2])
        def kernel(threads_arg, npt_arg):
            block_size = threads_arg * npt_arg

            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                lo: T.Tensor((N,), dtype),
                y: T.Tensor((N,), dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    lo_reg = T.alloc_fragment((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    T.copy(lo[bx * block_size : (bx + 1) * block_size], lo_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        k = i * npt_arg + j
                        x32 = T.cast(x_reg[k], "float32")
                        lo32 = T.cast(lo_reg[k], "float32")
                        r = T.max(x32, lo32)
                        # NaN propagation (PyTorch clamp_min):
                        # if x or lo is NaN -> output NaN.
                        r = T.if_then_else(T.isnan(lo32), lo32, r)
                        r = T.if_then_else(T.isnan(x32), x32, r)
                        x_reg[k] = T.Cast(dtype, r)
                    T.copy(x_reg, y[bx * block_size : (bx + 1) * block_size])

            return main

        return kernel

    # has_max only
    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            hi: T.Tensor((N,), dtype),
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                hi_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                T.copy(hi[bx * block_size : (bx + 1) * block_size], hi_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    x32 = T.cast(x_reg[k], "float32")
                    hi32 = T.cast(hi_reg[k], "float32")
                    r = T.min(x32, hi32)
                    # NaN propagation (PyTorch clamp_max):
                    # if x or hi is NaN -> output NaN.
                    r = T.if_then_else(T.isnan(hi32), hi32, r)
                    r = T.if_then_else(T.isnan(x32), x32, r)
                    x_reg[k] = T.Cast(dtype, r)
                T.copy(x_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class ClampTensorFwdKernel(ParametricUnaryKernel):
    """Tensor-bound clamp kernel.

    Computes ``y = clamp(x, lo, hi)``. ``forward`` takes the manifest shapes,
        broadcasts ``input`` / ``min`` / ``max`` to the output shape and flattens
        them; the PrimFunc works on length ``N_total``. ``has_min`` / ``has_max``
        select between the three forms used by the Tensor clamp, clamp_min, and
        clamp_max ops.
    """

    _DEFAULT_THREADS = 512

    def __init__(self, N_total, dtype, has_min, has_max, config=None, tune=False):
        if not (has_min or has_max):
            raise ValueError(
                "ClampTensorFwdKernel requires has_min or has_max to be True",
            )
        self.has_min = bool(has_min)
        self.has_max = bool(has_max)
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_clamp_tensor_kernel

    def _builder_args(self):
        return (self.has_min, self.has_max)

    def forward(self, x, lo=None, hi=None):
        self._require_cuda(x=x, lo=lo, hi=hi)
        out_shape = _broadcast_target(x, lo, hi)
        x_flat = _expand_flat(x, out_shape)
        if self.has_min and self.has_max:
            result = self._compiled_fn(
                x_flat, _expand_flat(lo, out_shape), _expand_flat(hi, out_shape)
            )
        elif self.has_min:
            result = self._compiled_fn(x_flat, _expand_flat(lo, out_shape))
        else:
            result = self._compiled_fn(x_flat, _expand_flat(hi, out_shape))
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result.reshape(out_shape)
