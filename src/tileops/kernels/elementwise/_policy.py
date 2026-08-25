"""Elementwise launch, output, and strategy helpers."""

import warnings
from dataclasses import dataclass

import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._dtype import (
    _fp8_accum_dtype_str,
    _fp8_needs_nonsaturating_cast,
    _is_fp8,
    _torch_dtype_nbytes,
)

_DEFAULT_THREADS = 128
_DIRECT_THREADS = 256
_BYTES_PER_THREAD = 16
_MIN_NUM_PER_THREAD = 4
_BOOL_OUTPUT_MAX_NPT = 4
_MAX_THREADS = 1024
_TARGET_BLOCKS = 256
_FP8_NPT = 16
_BOOL_STORAGE_DTYPE = "int8"


def default_launch_config(
    *,
    strategy: str,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    n_total: int | None,
    stores_bool: bool = True,
) -> dict:
    """Return the default launch config for one elementwise specialization."""
    # A direct block covers ``threads`` elements where a vectorized one covers
    # ``threads * num_per_thread``: the elements per block, not the thread count,
    # are what has to stay wide enough to keep the memory pipe busy.
    threads = _DIRECT_THREADS if strategy == "direct" else _DEFAULT_THREADS
    if _is_fp8(input_dtype):
        return {"strategy": strategy, "threads": threads, "num_per_thread": _FP8_NPT}

    elem_bytes = _torch_dtype_nbytes(input_dtype)
    npt = max(_MIN_NUM_PER_THREAD, _BYTES_PER_THREAD // elem_bytes)

    if output_dtype == torch.bool and stores_bool:
        capped = min(npt, _BOOL_OUTPUT_MAX_NPT)
        if strategy != "direct":  # a direct block spans `threads` whatever npt says
            threads = min(_MAX_THREADS, threads * npt // capped)
        npt = capped
    elif _torch_dtype_nbytes(output_dtype) < elem_bytes:
        npt *= 2

    while (
        n_total is not None
        and strategy != "direct"
        and npt > _MIN_NUM_PER_THREAD
        and n_total < threads * npt * _TARGET_BLOCKS
    ):
        npt //= 2
    return {"strategy": strategy, "threads": threads, "num_per_thread": npt}


def _store_bool_as_int8(op_func, arity: int):
    if arity == 1:

        def wrapped(x):
            return T.if_then_else(
                op_func(x),
                T.cast(1, _BOOL_STORAGE_DTYPE),
                T.cast(0, _BOOL_STORAGE_DTYPE),
            )
    else:

        def wrapped(a, b):
            return T.if_then_else(
                op_func(a, b),
                T.cast(1, _BOOL_STORAGE_DTYPE),
                T.cast(0, _BOOL_STORAGE_DTYPE),
            )

    return wrapped


def _store_unary_bool_as_int8(op_func):
    return _store_bool_as_int8(op_func, arity=1)


def _store_binary_bool_as_int8(op_func):
    return _store_bool_as_int8(op_func, arity=2)


@dataclass(frozen=True)
class ElementwiseOutputPlan:
    logical_dtype: torch.dtype
    kernel_output_dtype: str | None
    post_cast_dtype: torch.dtype | None = None
    bool_via_int8: bool = False


def elementwise_output_plan(
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None = None,
    *,
    strategy: str | None = None,
    bool_storage: bool = False,
) -> ElementwiseOutputPlan:
    post_cast_dtype = None
    logical_dtype = declared_output_dtype or input_dtype
    if (
        declared_output_dtype is None
        and _is_fp8(input_dtype)
        and _fp8_needs_nonsaturating_cast(input_dtype)
    ):
        logical_dtype, post_cast_dtype = torch.float16, input_dtype

    bool_via_int8 = (
        bool_storage and declared_output_dtype == torch.bool and strategy == "register_copy"
    )
    if bool_via_int8:
        kernel_output_dtype = _BOOL_STORAGE_DTYPE
    elif post_cast_dtype is not None:
        kernel_output_dtype = _fp8_accum_dtype_str()
    else:
        kernel_output_dtype = Kernel.dtype_to_str(logical_dtype)

    return ElementwiseOutputPlan(logical_dtype, kernel_output_dtype, post_cast_dtype, bool_via_int8)


def _bool_output_needs_scalar(
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None,
) -> bool:
    return declared_output_dtype == torch.bool and input_dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
    )


def _get_fp8_output_dtypes(dtype: torch.dtype):
    if _is_fp8(dtype) and _fp8_needs_nonsaturating_cast(dtype):
        return dtype, torch.float16
    return None, dtype


def _wrap_fp8_accumulation(base_op, dtype, dtype_str, arity=1):
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
    elif arity == 1:

        def fp8_accum_op(x):
            return T.Cast(dtype_str, base_op(T.cast(x, accum)))
    else:

        def fp8_accum_op(a, b):
            return T.Cast(dtype_str, base_op(T.cast(a, accum), T.cast(b, accum)))

    return fp8_accum_op


def _validate_strategy(requested: str | None, strategies: list[str]) -> None:
    if requested is not None and requested not in strategies:
        raise ValueError(f"Unknown strategy '{requested}', expected one of {strategies}")


def _warn_direct_override(
    requested: str | None, kernel_name: str, dtype: torch.dtype | None = None
) -> None:
    if requested is None or requested == "direct":
        return
    dtype_msg = "dtype=torch.bool" if dtype is None else f"dtype={dtype} with torch.bool output"
    warnings.warn(
        f"{kernel_name}: {dtype_msg} requires strategy='direct' "
        f"(TileLang cannot lower vectorised boolx<N>); "
        f"overriding requested strategy={requested!r}.",
        RuntimeWarning,
        stacklevel=3,
    )


def choose_unary_strategy(
    *,
    requested: str | None,
    strategies: list[str],
    default_strategy: str,
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None,
) -> str:
    _validate_strategy(requested, strategies)
    if input_dtype == torch.bool:
        _warn_direct_override(requested, "UnaryKernel")
        return "direct"
    if _bool_output_needs_scalar(input_dtype, declared_output_dtype):
        _warn_direct_override(requested, "UnaryKernel", input_dtype)
        return "direct"
    if requested is None and _is_fp8(input_dtype):
        return "explicit_parallel"
    return requested or default_strategy


def choose_binary_strategy(
    *,
    requested: str | None,
    strategies: list[str],
    default_strategy: str,
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None,
    same_shape: bool,
) -> str:
    _validate_strategy(requested, strategies)
    if input_dtype == torch.bool:
        _warn_direct_override(requested, "BinaryKernel")
        return "direct"
    if _bool_output_needs_scalar(input_dtype, declared_output_dtype):
        _warn_direct_override(requested, "BinaryKernel", input_dtype)
        return "direct"
    if requested == "register_copy" and not same_shape:
        return "explicit_parallel"
    return requested or ("register_copy" if same_shape else default_strategy)
