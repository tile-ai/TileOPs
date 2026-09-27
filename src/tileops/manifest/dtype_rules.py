"""The dtype registry: every dtype name a manifest may write, with its bits per element.

This package depends on nothing beyond the standard library and PyYAML, so dtypes are names.
"""

from __future__ import annotations

__all__ = ["DTYPE_BITS"]

DTYPE_BITS: dict[str, int] = {
    "bool": 8,
    "uint8": 8,
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
    "float16": 16,
    "bfloat16": 16,
    "float32": 32,
    "float64": 64,
    "complex64": 64,
    "complex128": 128,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float8_e4m3": 8,
    "float8_e5m2fnuz": 8,
    "float8_e4m3fnuz": 8,
}
