"""Dtype and scalar coercion helpers for elementwise kernels."""

import math

import tilelang.language as T
import torch

# Bool has no vectorised CUDA type, so a bool result is stored one byte per element.
BOOL_STORAGE_DTYPE = "int8"


def log_for_output_precision(value, wide):
    """Return ``log(wide)`` computed to the precision *value*'s dtype can keep."""
    return T.log(wide) if value.dtype == "float32" else T.__log(wide)


_BITWISE_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


# The dtypes every elementwise kernel refuses.
_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


_LOGICAL_DTYPES = _BITWISE_DTYPES + _FLOAT_DTYPES


_BINARY_FULL_DTYPES = _BITWISE_DTYPES + (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


_BINARY_NO_BOOL_DTYPES = tuple(dt for dt in _BINARY_FULL_DTYPES if dt is not torch.bool)


def _torch_dtype_nbytes(dtype: torch.dtype) -> int:
    """Return the byte width of a torch dtype."""
    return torch.empty(0, dtype=dtype).element_size()


def _clamp_to_dtype_range(value, dtype: torch.dtype):
    """Normalize *value* into the storage representation of *dtype*."""
    if dtype == torch.bool:
        return 1 if bool(value) else 0
    if dtype in _BITWISE_DTYPES:
        if isinstance(value, float) and math.isinf(value):
            iinfo = torch.iinfo(dtype)
            return iinfo.max if value > 0 else iinfo.min
        if (
            dtype == torch.uint8
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value < 0
        ):
            return value & 0xFF
        return int(value)
    fvalue = float(value)
    if math.isnan(fvalue):
        return fvalue
    finfo = torch.finfo(dtype)
    if math.isinf(fvalue):
        return fvalue
    return max(finfo.min, min(finfo.max, fvalue))
