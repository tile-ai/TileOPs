"""TIR helpers for unpacking low-precision integers from packed storage.

Vendored verbatim from the quantize package of tile-ai/tilelang (file
``quantization.py``, commit ``afcebed1``; the package was removed upstream
in tilelang#2761).
"""

import tilelang  # noqa: F401  # registers the vendored "tvm" module on import

try:
    from tvm import tirx as tir  # pinned tilelang names its tir module "tirx"
except ImportError:
    from tvm import tir

__all__ = ["UINT4_TO_FP16_LOP3_SOURCE", "_tir_packed_to_unsigned_convert"]


# Vendored from tile-ai/tilelang's former ``tilelang.quantize.lop3`` module at
# the same commit documented above.  Newer TileLang releases removed that
# Python package, but W4 decode still needs the four-instruction LOP3 sequence
# rather than eight scalar shifts and casts.
UINT4_TO_FP16_LOP3_SOURCE = r"""
template <typename T1, typename T2>
__device__ __forceinline__ void decode_i4u_to_f16(
    T1* packed, T2* decoded, const int N = 8) {
  uint* half2 = reinterpret_cast<uint*>(decoded);
  constexpr uint kLut = (0xf0 & 0xcc) | 0xaa;
  constexpr uint kBottomMask = 0x000f000f;
  constexpr uint kFp16Magic = 0x64006400;
  const uint values = *reinterpret_cast<uint*>(packed);
#pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(half2[i])
                 : "r"(values >> (4 * i)), "n"(kBottomMask),
                   "n"(kFp16Magic), "n"(kLut));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(half2[i])
                 : "r"(half2[i]), "r"(kFp16Magic));
  }
}
"""


def _tir_packed_to_unsigned_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        mask = tir.const((1 << nbit) - 1, storage_dtype)
        return ((val >> (pos * nbit).astype(storage_dtype)) & mask).astype(dtype)

    return f_convert
