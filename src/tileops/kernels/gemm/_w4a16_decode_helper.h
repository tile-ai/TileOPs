#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

// One dequantized FP16 weight from a 32-bit packed word.
//
// `word`  four packed bytes, eight INT4 weights.
// `bias`  1024 + zero point. The FP16 pattern 0x6400 ORed with an unsigned
//         nibble reads back as 1024 + q, so the unpack needs no int-to-float
//         convert and this subtraction is exact.
// `scale` the group's scale.
// `j`     which LOP3 pair, 0..3: the LOP3 returns nibbles j and j+4.
// `v`     which half of that pair, 0 or 1.
//
// Returns one half rather than the packed pair: splitting a uint32 in TileLang
// lowers to shift, mask, truncate and reassemble, which nvcc does not fold.
// `j` and `v` are compile-time constants at every call site, so nvcc shares the
// LOP3 and the subtract across the calls that decode one word.
// `repack_w4a16_weight` arranges the nibble order this pairing needs.
__device__ __forceinline__ cutlass::half_t tileops_w4a16_dequant_word(
    unsigned int word, cutlass::half_t bias, cutlass::half_t scale, int j, int v) {
  static constexpr unsigned int kImmLut = (0xf0 & 0xcc) | 0xaa;  // (a & b) | c
  unsigned int h;
  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(h)
      : "r"(word >> (4 * j)), "n"(0x000f000fu), "n"(0x64006400u), "n"(kImmLut));
  const __half2 bias2 = __half2half2(*reinterpret_cast<const __half *>(&bias));
  const __half2 scale2 = __half2half2(*reinterpret_cast<const __half *>(&scale));
  const __half2 out =
      __hmul2(__hsub2(*reinterpret_cast<const __half2 *>(&h), bias2), scale2);
  const __half r = v == 0 ? __low2half(out) : __high2half(out);
  return *reinterpret_cast<const cutlass::half_t *>(&r);
}
