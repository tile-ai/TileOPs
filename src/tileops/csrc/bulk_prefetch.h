#pragma once

namespace tileops {

// Asks the TMA unit to bring `bytes` (a multiple of 16) of global memory,
// starting `offset` bytes past `base`, into L2. Nothing waits for it: a later
// load of those bytes finds them in L2 instead of DRAM.
__device__ __forceinline__ void bulk_prefetch_l2(const void* base,
                                                 long long offset,
                                                 unsigned bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(
                   static_cast<const char*>(base) + offset),
               "r"(bytes)
               : "memory");
}

}  // namespace tileops
