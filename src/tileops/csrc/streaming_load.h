#pragma once

#include <cstdint>

namespace tl {

// Copies 16 bytes of global memory that the kernel reads exactly once into dst,
// marked first for eviction from L1 and L2 so the stream does not displace the
// lines other data holds. The asm is not volatile: the source is read-only for
// the kernel's lifetime, and the compiler may issue several loads ahead.
__device__ __forceinline__ void tileops_load16_evict_first(void* dst, const void* src) {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  uint4 v;
  asm("ld.global.L1::evict_first.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(src), "l"(policy));
  __builtin_memcpy(dst, &v, 16);
}

}  // namespace tl
