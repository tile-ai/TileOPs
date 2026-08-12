// Named-barrier arrive, which TileLang does not expose as a primitive.
//
// `bar.arrive` lets one warpgroup signal a hardware barrier without blocking on
// it, which the warp-specialized attention kernels use to hand a stage over.

#pragma once

namespace tl {

__device__ __forceinline__ void tileops_barrier_arrive_named(int barrier_id,
                                                             int thread_count) {
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(thread_count));
}

}  // namespace tl
