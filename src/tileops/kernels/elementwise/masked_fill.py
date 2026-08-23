"""Masked-fill kernels: scalar fill value and tensor fill value."""

import functools

import tilelang
import tilelang.language as T
import torch

from ._base import (
    _BITWISE_DTYPES,
    _FLOAT_DTYPES,
    ParametricUnaryKernel,
    _broadcast_target,
    _clamp_to_dtype_range,
    _expand_flat,
)

__all__ = [
    "MaskedFillFwdKernel",
    "MaskedFillTensorValueFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_masked_fill_kernel(
    N, dtype, fill_value, output_dtype=None, is_fp8=False, threads=256, npt=8
):
    """Build masked_fill kernel: out = mask ? fill_value : x.

    ``MaskedFillFwdKernel.forward`` packs the bool mask as uint8 so that T.copy
        can perform vectorized loads (TileLang does not vectorize bool tensors).
        Each uint8 element is 0 or 1; the kernel loads it into a register
        fragment and unpacks per-element with a != 0 comparison.

        For non-fp8 dtypes, writes the result back into the x register fragment
        (in-place) to reduce register pressure and avoid a third data-typed
        fragment allocation.

        For e5m2, the PrimFunc outputs fp16 so ``forward`` can do a
        non-saturating cast to e5m2.
    """
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    if is_fp8:

        @tilelang.jit(out_idx=[2])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                mask: T.Tensor((N,), "uint8"),
                out: T.Tensor((N,), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            fv = T.cast(fill_value, out_dtype)
                            x_val = T.Cast(out_dtype, x[idx])
                            out[idx] = T.if_then_else(
                                mask[idx] != T.cast(0, "uint8"),
                                fv,
                                x_val,
                            )

            return main
    else:

        @tilelang.jit(out_idx=[2])
        def kernel(threads_arg, npt_arg):
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


class MaskedFillFwdKernel(ParametricUnaryKernel):
    """MaskedFill: out = mask ? fill_value : x.

    Supports the PyTorch ``Tensor.masked_fill(mask, value: Number)`` dtype
    union of integer and floating-point input dtypes, plus bool: bool storage
    is reinterpreted as uint8 here, because that is this backend's requirement
    rather than part of the op's semantics.
    """

    _DEFAULT_THREADS = 512
    SUPPORTED_DTYPES = _BITWISE_DTYPES[1:] + _FLOAT_DTYPES  # uint8/intN + fp16/bf16/fp32

    def __init__(self, N_total, dtype, fill_value, config=None, tune=False):
        self._raw_fill_value = fill_value
        super().__init__(N_total, dtype, config=config, tune=tune)

    def _post_init_params(self):
        self.fill_value = _clamp_to_dtype_range(self._raw_fill_value, self.output_dtype)

    @staticmethod
    def _builder_fn():
        return _make_masked_fill_kernel

    def _builder_args(self):
        return (self.fill_value,)

    def forward(self, x, mask):
        self._require_cuda(x=x, mask=mask)
        out_shape = _broadcast_target(x, mask)
        as_bool = x.dtype == torch.bool
        if as_bool:
            x = x.view(torch.uint8)
        if mask.dtype == torch.bool:
            mask = mask.view(torch.uint8)
        result = self._compiled_fn(
            _expand_flat(x, out_shape), _expand_flat(mask, out_shape)
        ).reshape(out_shape)
        return result.view(torch.bool) if as_bool else result


@functools.lru_cache(maxsize=32)
def _make_masked_fill_tensor_value_kernel(
    N, dtype, output_dtype=None, is_fp8=False, threads=256, npt=8
):
    """Build masked_fill kernel with a 0-dim Tensor fill value.

    Inputs (all flat, length N, broadcast and flattened by
        ``MaskedFillTensorValueFwdKernel.forward``):
            x: data tensor (length N).
            mask: bool mask packed as uint8 (length N).
            value: scalar fill value carried as a length-1 tensor (``forward``
                reshapes the 0-dim Tensor to $[1]$).

        Output:
            out: ``out[i] = value[0] if mask[i] else x[i]``.
    """
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    if is_fp8:

        @tilelang.jit(out_idx=[3])
        def kernel(threads_arg, npt_arg):
            @T.prim_func
            def main(
                x: T.Tensor((N,), dtype),
                mask: T.Tensor((N,), "uint8"),
                value: T.Tensor((1,), dtype),
                out: T.Tensor((N,), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                    fv = T.Cast(out_dtype, value[0])
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = (bx * threads_arg + i) * npt_arg + j
                        if idx < N:
                            x_val = T.Cast(out_dtype, x[idx])
                            out[idx] = T.if_then_else(
                                mask[idx] != T.cast(0, "uint8"),
                                fv,
                                x_val,
                            )

            return main

        return kernel

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
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


class MaskedFillTensorValueFwdKernel(ParametricUnaryKernel):
    """MaskedFill kernel with 0-dim Tensor fill value.

    Computes ``out = mask ? value : x``. ``forward`` broadcasts ``input`` and
        ``mask`` to the output shape, flattens them, packs the mask as uint8, and
        reshapes the 0-dim ``value`` to a length-1 tensor; the PrimFunc works on
        length ``N_total``. Bool storage is
        reinterpreted as uint8 here, being this backend's requirement rather than
        part of the op's semantics.
    """

    _DEFAULT_THREADS = 512
    SUPPORTED_DTYPES = _BITWISE_DTYPES[1:] + _FLOAT_DTYPES  # uint8/intN + fp16/bf16/fp32

    @staticmethod
    def _builder_fn():
        return _make_masked_fill_tensor_value_kernel

    def forward(self, x, mask, value):
        self._require_cuda(x=x, mask=mask, value=value)
        out_shape = _broadcast_target(x, mask)
        as_bool = x.dtype == torch.bool
        if as_bool:
            x = x.view(torch.uint8)
        # A 0-d scalar is the semantic form; this kernel reads it from a
        # length-one buffer.
        value = value.contiguous().reshape(1)
        if as_bool:
            value = value.view(torch.uint8)
        if mask.dtype == torch.bool:
            mask = mask.view(torch.uint8)
        result = self._compiled_fn(_expand_flat(x, out_shape), _expand_flat(mask, out_shape), value)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        result = result.reshape(out_shape)
        return result.view(torch.bool) if as_bool else result
