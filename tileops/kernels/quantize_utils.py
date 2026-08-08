"""TIR helpers for unpacking low-precision integers from packed storage.

Vendored from the quantize package of tile-ai/tilelang (file
``quantization.py``) at commit ``afcebed1`` — identical at ``65dbc98``, the
last commit before the package's removal in tilelang#2761. Only the
unsigned-unpack helper needed by the w4a16 GEMM kernel is vendored; behavior
is unchanged from upstream.
"""

import tilelang  # noqa: F401  # registers the vendored "tvm" module on import

try:
    from tvm import tirx as tir  # pinned tilelang names its tir module "tirx"
except ImportError:
    from tvm import tir

__all__ = ["_tir_packed_to_unsigned_convert"]


def _tir_packed_to_unsigned_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        mask = tir.const((1 << nbit) - 1, storage_dtype)
        return ((val >> (pos * nbit).astype(storage_dtype)) & mask).astype(dtype)

    return f_convert
