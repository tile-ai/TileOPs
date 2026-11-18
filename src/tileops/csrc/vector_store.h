#pragma once

namespace tileops {

// Stores the 16 bytes at src to global dst, which is 16-byte aligned, as one
// 16-byte store with the default cache policy. A C++ uint4 assignment, as in
// tileops::store16, can compile to four 4-byte stores; the inline st.global.v4
// cannot.
__device__ __forceinline__ void store16_global(void* dst, const void* src) {
  uint4 v;
  __builtin_memcpy(&v, src, 16);
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

}  // namespace tileops
