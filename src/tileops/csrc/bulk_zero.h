#pragma once

#include <cstdint>

namespace tileops {

// Zeroes chunks [from, to) of the calling CTA's share of `buf`: chunk k is the
// `bytes`-byte block gridDim.x * k + blockIdx.x, copied from the zeroed shared
// buffer `zeros` by bulk stores. The stores are evict-first in L2, so they do
// not displace the lines the kernel reuses. One thread calls it; the caller
// waits for the stores with `cp.async.bulk.wait_group`.
__device__ __forceinline__ void bulk_zero_chunks(void* buf, const void* zeros,
                                                 unsigned bytes, long long from,
                                                 long long to) {
  unsigned src = static_cast<unsigned>(__cvta_generic_to_shared(zeros));
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  for (long long k = from; k < to; ++k) {
    char* dst = static_cast<char*>(buf) +
                (k * gridDim.x + blockIdx.x) * static_cast<long long>(bytes);
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], "
        "[%1], "
        "%2, %3;" ::"l"(dst),
        "r"(src), "r"(bytes), "l"(policy)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    // Bound the groups in flight; the source never changes, so this waits on
    // nothing the stores need.
    if ((k & 7) == 7)
      asm volatile("cp.async.bulk.wait_group.read 8;" ::: "memory");
  }
}

}  // namespace tileops
