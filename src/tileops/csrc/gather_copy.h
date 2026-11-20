#pragma once

#include <cstdint>

namespace tileops {

// Loads B bytes (4, 8 or 16) from global src into dst, evict-last in L2: a
// gathered table row that later lookups may read again stays ahead of streamed
// output. A warp's 16-byte vectors cover whole lines of a row once, so they
// bypass L1 (.cg); 4- and 8-byte vectors belong to rows narrower than a line,
// whose other sectors later lookups in the block may read, so they cache in L1
// (.ca). Volatile, so every load a thread issues stays ahead of its stores.
template <int B>
__device__ __forceinline__ void load_evict_last(void* dst, const void* src) {
  static_assert(B == 4 || B == 8 || B == 16, "a load moves 4, 8 or 16 bytes");
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  if constexpr (B == 16) {
    uint4 v;
    asm volatile(
        "ld.global.cg.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(src), "l"(policy));
    *static_cast<uint4*>(dst) = v;
  } else if constexpr (B == 8) {
    uint2 v;
    asm volatile("ld.global.ca.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(src), "l"(policy));
    *static_cast<uint2*>(dst) = v;
  } else {
    unsigned v;
    asm volatile("ld.global.ca.L2::cache_hint.u32 %0, [%1], %2;"
                 : "=r"(v)
                 : "l"(src), "l"(policy));
    *static_cast<unsigned*>(dst) = v;
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
