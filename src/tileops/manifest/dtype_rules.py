"""The dtype registry: every dtype name a manifest may write, with its bits per element and
its category.

This package depends on nothing beyond the standard library and PyYAML, so dtypes are names.
"""

from __future__ import annotations

__all__ = ["DTYPE_BITS", "DTYPE_CATEGORY", "FLOAT8_DTYPES"]

# name: (bits per element, category)
_DTYPES: dict[str, tuple[int, str]] = {
    "bool": (8, "bool"),
    "uint8": (8, "int"),
    "int8": (8, "int"),
    "int16": (16, "int"),
    "int32": (32, "int"),
    "int64": (64, "int"),
    "float16": (16, "float"),
    "bfloat16": (16, "float"),
    "float32": (32, "float"),
    "float64": (64, "float"),
    "complex64": (64, "complex"),
    "complex128": (128, "complex"),
    "float8_e4m3fn": (8, "float"),
    "float8_e5m2": (8, "float"),
    "float8_e4m3": (8, "float"),
    "float8_e5m2fnuz": (8, "float"),
    "float8_e4m3fnuz": (8, "float"),
}

DTYPE_BITS: dict[str, int] = {name: bits for name, (bits, _) in _DTYPES.items()}
# 'bool', 'int', 'float' or 'complex'.
DTYPE_CATEGORY: dict[str, str] = {name: kind for name, (_, kind) in _DTYPES.items()}
FLOAT8_DTYPES: frozenset[str] = frozenset(
    name for name, (bits, kind) in _DTYPES.items() if bits == 8 and kind == "float"
)
