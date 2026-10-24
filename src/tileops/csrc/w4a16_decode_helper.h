#pragma once

#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <cstring>

namespace tileops {

__device__ __forceinline__ __half2 w4a16_half2_from_bits(unsigned int bits) {
  __half2 value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

// One zero-centered FP16 weight from a 32-bit packed word.
//
// `word`  four packed bytes, eight INT4 weights.
// `bias`  1024 + zero point. The FP16 pattern 0x6400 ORed with an unsigned
//         nibble reads back as 1024 + q, so the unpack needs no int-to-float
//         convert and this subtraction is exact.
// `j`     which LOP3 pair, 0..3: the LOP3 returns nibbles j and j+4.
// `v`     which half of that pair, 0 or 1.
//
// Returns one half rather than the packed pair: splitting a uint32 in TileLang
// lowers to shift, mask, truncate and reassemble, which nvcc does not fold.
// `j` and `v` are compile-time constants at every call site, so nvcc shares the
// LOP3 and the subtract across the calls that decode one word.
// `repack_w4a16_weight` arranges the nibble order this pairing needs.
__device__ __forceinline__ cutlass::half_t w4a16_dequant_word(
    unsigned int word, cutlass::half_t bias, int j, int v) {
  static constexpr unsigned int kImmLut = (0xf0 & 0xcc) | 0xaa;  // (a & b) | c
  unsigned int h;
  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(h)
      : "r"(word >> (4 * j)), "n"(0x000f000fu), "n"(0x64006400u), "n"(kImmLut));
  const __half2 bias2 = __half2half2(__ushort_as_half(bias.raw()));
  const __half2 out = __hsub2(w4a16_half2_from_bits(h), bias2);
  const __half r = v == 0 ? __low2half(out) : __high2half(out);
  return cutlass::half_t(r);
}

// `w4a16_dequant_word` with one shift a word instead of three: same arguments,
// same result.
//
// Pairs 1 and 3 are masked where they sit, four bits up, so they read back as
// 1024 + 16q; one FMA by 1/16 minus (64 + zero) centers them exactly. Pairs 2
// and 3 read the word shifted by eight, a shift nvcc shares across both.
__device__ __forceinline__ cutlass::half_t w4a16_dequant_word_one_shift(
    unsigned int word, cutlass::half_t bias, int j, int v) {
  static constexpr unsigned int kImmLut = (0xf0 & 0xcc) | 0xaa;  // (a & b) | c
  const unsigned int src = j < 2 ? word : word >> 8;
  const __half bias_half = __ushort_as_half(bias.raw());
  unsigned int h;
  __half2 out;
  if (j % 2 == 0) {
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h)
        : "r"(src), "n"(0x000f000fu), "n"(0x64006400u), "n"(kImmLut));
    out = __hsub2(w4a16_half2_from_bits(h), __half2half2(bias_half));
  } else {
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h)
        : "r"(src), "n"(0x00f000f0u), "n"(0x64006400u), "n"(kImmLut));
    // (1024 + 16q) / 16 - (64 + zero); 960 = 1024 - 1024 / 16.
    const __half2 high_bias =
        __half2half2(__hsub(bias_half, __float2half(960.0f)));
    out = __hfma2(w4a16_half2_from_bits(h), __float2half2_rn(0.0625f),
                  __hneg2(high_bias));
  }
  const __half r = v == 0 ? __low2half(out) : __high2half(out);
  return cutlass::half_t(r);
}

}  // namespace tileops
