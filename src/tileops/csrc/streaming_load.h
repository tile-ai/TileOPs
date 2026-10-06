#pragma once

#include <cstdint>

namespace tl {

// Loads 16 bytes of read-once global memory into dst, evict-first in L1 and L2.
// Not volatile: the source is read-only for the kernel, so loads may issue
// ahead.
__device__ __forceinline__ void tileops_load16_evict_first(void* dst,
                                                           const void* src) {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  uint4 v;
  asm("ld.global.L1::evict_first.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], "
      "%5;"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(src), "l"(policy));
  __builtin_memcpy(dst, &v, 16);
}

// Loads one element of read-once global memory, evict-first in L2. Not
// volatile, as above. T is 4 or 2 bytes wide.
template <typename T>
__device__ __forceinline__ T tileops_load_evict_first(const T* src) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 2, "a 4- or 2-byte element");
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  T v;
  if constexpr (sizeof(T) == 4) {
    uint32_t bits;
    asm("ld.global.nc.L2::cache_hint.b32 %0, [%1], %2;"
        : "=r"(bits)
        : "l"(src), "l"(policy));
    __builtin_memcpy(&v, &bits, 4);
  } else {
    uint16_t bits;
    asm("ld.global.nc.L2::cache_hint.b16 %0, [%1], %2;"
        : "=h"(bits)
        : "l"(src), "l"(policy));
    __builtin_memcpy(&v, &bits, 2);
  }
  return v;
}

// Loads one float that every block of a kernel reads, evict-last in L1 so that
// later blocks on the same SM find it there rather than queueing on the one L2
// line.
__device__ __forceinline__ float tileops_load_f32_evict_last(const float* src) {
  float v;
  asm volatile("ld.global.nc.L1::evict_last.f32 %0, [%1];"
               : "=f"(v)
               : "l"(src));
  return v;
}

// Loads 16 bytes of global memory into dst with the default cache policy.
__device__ __forceinline__ void tileops_load16(void* dst, const void* src) {
  uint4 v = *reinterpret_cast<const uint4*>(src);
  __builtin_memcpy(dst, &v, 16);
}

}  // namespace tl
