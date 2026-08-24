"""Output storage planning for elementwise kernels."""

from dataclasses import dataclass

import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._dtype import _fp8_accum_dtype_str, _fp8_needs_nonsaturating_cast, _is_fp8

#: What a bool result is written through where the kernel can choose. int8 is the same
#: width as bool and 0/1 are the same two byte patterns, so the bool view copies nothing.
_BOOL_STORAGE_DTYPE: str = "int8"


def _store_unary_bool_as_int8(op_func):
    """Wrap a unary predicate so its bool result lands as ``_BOOL_STORAGE_DTYPE``."""

    def wrapped(x):
        return T.if_then_else(
            op_func(x),
            T.cast(1, _BOOL_STORAGE_DTYPE),
            T.cast(0, _BOOL_STORAGE_DTYPE),
        )

    return wrapped


def _store_binary_bool_as_int8(op_func):
    """Wrap a binary predicate so its bool result lands as ``_BOOL_STORAGE_DTYPE``."""

    def wrapped(a, b):
        return T.if_then_else(
            op_func(a, b),
            T.cast(1, _BOOL_STORAGE_DTYPE),
            T.cast(0, _BOOL_STORAGE_DTYPE),
        )

    return wrapped


@dataclass(frozen=True)
class ElementwiseOutputPlan:
    """Storage decisions for one elementwise kernel specialization."""

    logical_dtype: torch.dtype
    kernel_output_dtype: str | None
    post_cast_dtype: torch.dtype | None = None
    bool_via_int8: bool = False

    @classmethod
    def for_unary(
        cls,
        input_dtype: torch.dtype,
        declared_output_dtype: torch.dtype | None,
        strategy: str,
    ) -> "ElementwiseOutputPlan":
        logical_dtype, post_cast_dtype = _logical_output_dtype(input_dtype, declared_output_dtype)
        bool_via_int8 = declared_output_dtype == torch.bool and strategy == "register_copy"
        return cls._from_parts(logical_dtype, post_cast_dtype, bool_via_int8)

    @classmethod
    def for_binary(
        cls,
        input_dtype: torch.dtype,
        declared_output_dtype: torch.dtype | None,
        strategy: str,
    ) -> "ElementwiseOutputPlan":
        logical_dtype, post_cast_dtype = _logical_output_dtype(input_dtype, declared_output_dtype)
        bool_via_int8 = declared_output_dtype == torch.bool and strategy == "register_copy"
        return cls._from_parts(logical_dtype, post_cast_dtype, bool_via_int8)

    @classmethod
    def for_fused_gated(cls, input_dtype: torch.dtype) -> "ElementwiseOutputPlan":
        logical_dtype, post_cast_dtype = _logical_output_dtype(input_dtype, None)
        return cls._from_parts(logical_dtype, post_cast_dtype, bool_via_int8=False)

    @classmethod
    def _from_parts(
        cls,
        logical_dtype: torch.dtype,
        post_cast_dtype: torch.dtype | None,
        bool_via_int8: bool,
    ) -> "ElementwiseOutputPlan":
        if bool_via_int8:
            kernel_output_dtype = _BOOL_STORAGE_DTYPE
        elif post_cast_dtype is not None:
            kernel_output_dtype = _fp8_accum_dtype_str()
        else:
            kernel_output_dtype = Kernel.dtype_to_str(logical_dtype)
        return cls(
            logical_dtype=logical_dtype,
            kernel_output_dtype=kernel_output_dtype,
            post_cast_dtype=post_cast_dtype,
            bool_via_int8=bool_via_int8,
        )


def _logical_output_dtype(
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None,
) -> tuple[torch.dtype, torch.dtype | None]:
    """Return the logical output dtype and optional post-cast dtype."""
    if (
        declared_output_dtype is None
        and _is_fp8(input_dtype)
        and _fp8_needs_nonsaturating_cast(input_dtype)
    ):
        return torch.float16, input_dtype
    return declared_output_dtype or input_dtype, None


def _bool_output_needs_scalar(
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None,
) -> bool:
    """Whether TileLang cannot lower this bool-output vectorized path."""
    return declared_output_dtype == torch.bool and input_dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
    )


def _get_fp8_output_dtypes(dtype: torch.dtype):
    """Return (fp8_output_dtype, kernel output dtype) for fp8 handling."""
    if _is_fp8(dtype) and _fp8_needs_nonsaturating_cast(dtype):
        return dtype, torch.float16
    return None, dtype


def _wrap_fp8_accumulation(base_op, dtype, dtype_str, arity=1):
    """Wrap an op function with fp8 accumulation logic if *dtype* is fp8."""
    if not _is_fp8(dtype):
        return base_op

    accum = _fp8_accum_dtype_str()

    if _fp8_needs_nonsaturating_cast(dtype):
        if arity == 1:

            def fp8_accum_op(x):
                return base_op(T.cast(x, accum))
        else:

            def fp8_accum_op(a, b):
                return base_op(T.cast(a, accum), T.cast(b, accum))

        return fp8_accum_op

    if arity == 1:

        def fp8_accum_op(x):
            return T.Cast(dtype_str, base_op(T.cast(x, accum)))
    else:

        def fp8_accum_op(a, b):
            return T.Cast(dtype_str, base_op(T.cast(a, accum), T.cast(b, accum)))

    return fp8_accum_op
