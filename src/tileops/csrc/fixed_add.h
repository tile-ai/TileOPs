#pragma once

namespace tileops {

// Atomically add `value` to the unsigned 64-bit integer at `address`. CUDA's
// atomicAdd takes `unsigned long long`, which `uint64_t` is not on LP64, so a
// TileLang `uint64` buffer reaches it through this cast.
__device__ __forceinline__ void atomic_add_u64(void* address,
                                               unsigned long long value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(address), value);
}

}  // namespace tileops
