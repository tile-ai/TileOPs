#pragma once

#include <cstdint>

namespace tl {

// Loads 16 bytes of read-once global memory into dst, evict-first in L1 and L2.
// Not volatile: the source is read-only for the kernel, so loads may issue ahead.
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
