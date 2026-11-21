#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace tileops {

// Pack adjacent codes in high-nibble-first order. Keep the subtraction and
// multiplication individually rounded before converting to a signed nibble.
__device__ __forceinline__ uint32_t pack_int4_group_word(uint32_t a, uint32_t b,
                                                         uint32_t c, uint32_t d,
                                                         float offset,
                                                         float gain) {
  const uint32_t words[4] = {a, b, c, d};
  uint32_t result = 0;
#pragma unroll
  for (int i = 3; i >= 0; --i) {
    const float lo = __half2float(__ushort_as_half(words[i]));
    const float hi = __half2float(__ushort_as_half(words[i] >> 16));
    const int first = __float2int_rn(__fmul_rn(__fsub_rn(lo, offset), gain));
    const int second = __float2int_rn(__fmul_rn(__fsub_rn(hi, offset), gain));
    asm("cvt.pack.sat.s4.s32.b32 %0, %1, %2, %0;"
        : "+r"(result)
        : "r"(first), "r"(second));
  }
  return result;
}

}  // namespace tileops
