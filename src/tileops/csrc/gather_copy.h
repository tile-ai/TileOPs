#pragma once

#include <cstdint>

namespace tileops {

// Starts a cp.async of B bytes (4, 8 or 16) from global src to shared dst,
// evict-last in L2: a gathered table row that later lookups may read again
// stays ahead of streamed output.
template <int B>
__device__ __forceinline__ void cp_async_evict_last(void* dst,
                                                    const void* src) {
  static_assert(B == 4 || B == 8 || B == 16, "cp.async moves 4, 8 or 16 bytes");
  unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  if constexpr (B == 16) {
    asm volatile(
        "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(
            shared),
        "l"(src), "l"(policy)
        : "memory");
  } else {
    asm volatile(
        "cp.async.ca.shared.global.L2::cache_hint [%0], [%1], %2, %3;" ::"r"(
            shared),
        "l"(src), "n"(B), "l"(policy)
        : "memory");
  }
}

// Stores the B bytes (4, 8 or 16) at src to global dst as streaming data
// (st.global.cs), which no later read of this kernel needs.
template <int B>
__device__ __forceinline__ void store_streaming(void* dst, const void* src) {
  if constexpr (B == 16) {
    uint4 v = *static_cast<const uint4*>(src);
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst),
                 "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
  } else if constexpr (B == 8) {
    uint2 v = *static_cast<const uint2*>(src);
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};" ::"l"(dst), "r"(v.x),
                 "r"(v.y)
                 : "memory");
  } else {
    static_assert(B == 4, "a streaming store moves 4, 8 or 16 bytes");
    unsigned v = *static_cast<const unsigned*>(src);
    asm volatile("st.global.cs.u32 [%0], %1;" ::"l"(dst), "r"(v) : "memory");
  }
}

}  // namespace tileops
