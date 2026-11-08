#pragma once

#include <cstdint>

namespace tileops {

// The low bf16 of the 4-byte pair at pair, as f32. A bf16 is the high half of
// the f32 it rounds, so widening it is exact.
__device__ __forceinline__ float widen_bf16_lo(const void* pair) {
  uint32_t bits;
  __builtin_memcpy(&bits, pair, 4);
  return __uint_as_float(bits << 16);
}

// The high bf16 of the 4-byte pair at pair, as f32.
__device__ __forceinline__ float widen_bf16_hi(const void* pair) {
  uint32_t bits;
  __builtin_memcpy(&bits, pair, 4);
  return __uint_as_float(bits & 0xffff0000u);
}

}  // namespace tileops
