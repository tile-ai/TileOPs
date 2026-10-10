#pragma once

namespace tileops {

// Lane 0 claims the next tile for its CTA; every lane of the warp gets it.
__device__ __forceinline__ int claim_tile(int* sched) {
  int t = 0;
  if ((threadIdx.x & 31) == 0) t = atomicAdd(sched, 1) + gridDim.x;
  return __shfl_sync(0xffffffffu, t, 0);
}

// Called once per CTA after all its claims: the last CTA out zeroes the
// counters, so every launch finds them zeroed.
__device__ __forceinline__ void retire(int* sched) {
  __threadfence();
  if (atomicAdd(sched + 1, 1) == gridDim.x - 1) {
    sched[0] = 0;
    sched[1] = 0;
  }
}

}  // namespace tileops
