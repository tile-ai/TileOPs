#pragma once

#include <cstdint>

namespace tileops {

// Loads the 4 bytes at `local`'s offset in the shared memory of CTA `rank` of
// this cluster.
__device__ __forceinline__ uint32_t cluster_load_u32(const void* local,
                                                     uint32_t rank) {
  uint32_t remote, value;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote)
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(local))),
                 "r"(rank));
  asm volatile("ld.shared::cluster.u32 %0, [%1];"
               : "=r"(value)
               : "r"(remote)
               : "memory");
  return value;
}

// Stores the 16 bytes at src to dst in global memory.
__device__ __forceinline__ void store16(void* dst, const void* src) {
  uint4 v;
  __builtin_memcpy(&v, src, 16);
  *reinterpret_cast<uint4*>(dst) = v;
}

}  // namespace tileops
