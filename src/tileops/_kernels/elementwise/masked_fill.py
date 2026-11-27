"""Masked-fill kernels: scalar fill value and tensor fill value."""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.elementwise._base import MultiInputElementwiseKernel
from tileops._kernels.elementwise._dtype import FLOAT_DTYPES, INT_DTYPES, clamp_to_dtype_range
from tileops._kernels.elementwise.call_spec import (
    ElementwiseCall,
    MaskedFillCall,
    MaskedFillFwdInterface,
    MaskedFillTensorValueFwdInterface,
)
from tileops._kernels.elementwise.where import select
from tileops._kernels.kernel_base import Entry

__all__ = [
    "MaskedFillFwdKernel",
    "MaskedFillTensorValueFwdKernel",
]

# Bool operands arrive in uint8 storage.
_MASKED_FILL_DTYPES = INT_DTYPES + FLOAT_DTYPES


def _bool_as_uint8(dtype, value):
    """The storage type these programs compute *dtype* in, and *value* in its value set."""
    if dtype != torch.bool:
        return dtype, value
    return torch.uint8, (None if value is None else int(bool(value)))


@target_cache(maxsize=32)
def _make_masked_fill_kernel(N, dtype, fill_value):
    """Build masked_fill kernel: out = mask ? fill_value : x, the bool mask passed as uint8
    (TileLang does not vectorize bool tensors); the result is written back into ``x``'s
    fragment."""

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
                    x_reg[k] = select(m_reg[k], T.cast(fill_value, dtype), x_reg[k])
                T.copy(x_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class MaskedFillFwdKernel(MultiInputElementwiseKernel, MaskedFillFwdInterface):
    """MaskedFill: out = mask ? fill_value : x; a bool input computes in uint8 storage."""

    SUPPORTED_DTYPES = _MASKED_FILL_DTYPES
    INPUTS = (("x", "tile"), ("mask", "mask"))

    @classmethod
    def refusal(cls, call) -> "str | None":
        return None if call.dtype == torch.bool else super().refusal(call)

    @classmethod
    def entry_for(cls, call: MaskedFillCall) -> Entry:
        dtype, value = _bool_as_uint8(call.dtype, call.value)
        return call, lambda: cls(call.n_total, dtype, value)

    def __init__(self, N_total, dtype, fill_value, config=None):
        self.fill_value = clamp_to_dtype_range(fill_value, dtype)
        super().__init__(N_total, dtype, config=config)

    _builder = staticmethod(_make_masked_fill_kernel)

    def _builder_args(self):
        return (self.fill_value,)

    def _element(self, x, mask):
        return select(mask, T.cast(self.fill_value, self.dtype_str), x)

    def forward(self, x, mask):
        return self._run(x=x, mask=mask)


@target_cache(maxsize=32)
def _make_masked_fill_tensor_value_kernel(N, dtype):
    """Build masked_fill kernel with a 0-dim Tensor fill value, passed as a length-one buffer
    and read once before the element loop."""

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
                    x_reg[k] = select(m_reg[k], fv, x_reg[k])
                T.copy(x_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class MaskedFillTensorValueFwdKernel(
    MultiInputElementwiseKernel, MaskedFillTensorValueFwdInterface
):
    """MaskedFill: out = mask ? value : x with a 0-dim Tensor value; a bool input computes in
    uint8 storage."""

    SUPPORTED_DTYPES = _MASKED_FILL_DTYPES
    INPUTS = (("x", "tile"), ("mask", "mask"), ("value", "value"))

    @classmethod
    def refusal(cls, call) -> "str | None":
        return None if call.dtype == torch.bool else super().refusal(call)

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        dtype, _ = _bool_as_uint8(call.dtype, None)
        return call, lambda: cls(call.n_total, dtype)

    _builder = staticmethod(_make_masked_fill_tensor_value_kernel)

    def _element(self, x, mask, value):
        return select(mask, value, x)

    def forward(self, x, mask, value):
        return self._run(x=x, mask=mask, value=value)
