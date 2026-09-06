"""Masked-fill kernels: scalar fill value and tensor fill value."""

import functools

import tilelang
import tilelang.language as T

from ._base import MultiInputElementwiseKernel
from ._dtype import _BITWISE_DTYPES, _FLOAT_DTYPES, _clamp_to_dtype_range

__all__ = [
    "MaskedFillFwdKernel",
    "MaskedFillTensorValueFwdKernel",
]

# uint8/intN + fp16/bf16/fp32. bool operands arrive in uint8 storage.
_MASKED_FILL_DTYPES = _BITWISE_DTYPES[1:] + _FLOAT_DTYPES


@functools.lru_cache(maxsize=32)
def _make_masked_fill_kernel(N, dtype, fill_value, threads=256, npt=8):
    """Build masked_fill kernel: out = mask ? fill_value : x.

    ``MaskedFillFwdKernel.forward`` packs the bool mask as uint8 so that T.copy
    can perform vectorized loads (TileLang does not vectorize bool tensors).
    Each uint8 element is 0 or 1; the kernel loads it into a register fragment
    and unpacks per-element with a != 0 comparison.

    The result is written back into ``x``'s register fragment rather than a
    third data-typed one.
    """

    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            mask: T.Tensor((N,), "uint8"),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                m_reg = T.alloc_fragment((block_size,), "uint8")
                x_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(mask[bx * block_size : (bx + 1) * block_size], m_reg)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    fv = T.cast(fill_value, dtype)
                    x_reg[k] = T.if_then_else(
                        m_reg[k] != T.cast(0, "uint8"),
                        fv,
                        x_reg[k],
                    )
                T.copy(x_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class MaskedFillFwdKernel(MultiInputElementwiseKernel):
    """MaskedFill: out = mask ? fill_value : x.

    Supports the PyTorch ``Tensor.masked_fill(mask, value: Number)`` dtype
    union of integer and floating-point input dtypes, plus bool: bool storage
    is reinterpreted as uint8 here, because that is this backend's requirement
    rather than part of the op's semantics.
    """

    DEFAULT_THREADS = 512
    SUPPORTED_DTYPES = _MASKED_FILL_DTYPES
    INPUTS = (("x", "tile"), ("mask", "mask"))

    def __init__(self, N_total, dtype, fill_value, config=None, tune=False):
        self.fill_value = _clamp_to_dtype_range(fill_value, dtype)
        super().__init__(N_total, dtype, config=config, tune=tune)

    @staticmethod
    def _builder_fn():
        return _make_masked_fill_kernel

    def _builder_args(self):
        return (self.fill_value,)

    def forward(self, x, mask):
        return self._run(x=x, mask=mask)


@functools.lru_cache(maxsize=32)
def _make_masked_fill_tensor_value_kernel(N, dtype, threads=256, npt=8):
    """Build masked_fill kernel with a 0-dim Tensor fill value.

    Inputs are flat and of length *N*, except ``value``, which carries the
    0-dim scalar in a length-one buffer. It is read once before the element
    loop, not per element. The result is written back into ``x``'s fragment.
    """

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            mask: T.Tensor((N,), "uint8"),
            value: T.Tensor((1,), dtype),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                m_reg = T.alloc_fragment((block_size,), "uint8")
                x_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(mask[bx * block_size : (bx + 1) * block_size], m_reg)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                fv = value[0]
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    x_reg[k] = T.if_then_else(
                        m_reg[k] != T.cast(0, "uint8"),
                        fv,
                        x_reg[k],
                    )
                T.copy(x_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class MaskedFillTensorValueFwdKernel(MultiInputElementwiseKernel):
    """MaskedFill kernel with 0-dim Tensor fill value.

    Computes ``out = mask ? value : x``. Bool storage is reinterpreted as uint8
    here, being this backend's requirement rather than part of the op's
    semantics.
    """

    DEFAULT_THREADS = 512
    SUPPORTED_DTYPES = _MASKED_FILL_DTYPES
    INPUTS = (("x", "tile"), ("mask", "mask"), ("value", "value"))

    @staticmethod
    def _builder_fn():
        return _make_masked_fill_tensor_value_kernel

    def forward(self, x, mask, value):
        return self._run(x=x, mask=mask, value=value)
