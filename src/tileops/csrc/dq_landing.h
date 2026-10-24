#pragma once

// Per-head counts that let the dQ landing kernel, launched as a programmatic
// dependent of the backward, land a head as soon as every addition into its
// f32 accumulator has landed, while the backward still runs other heads.

namespace tileops {

// The calling thread's bulk additions into one head's accumulator have
// completed: count them.
__device__ __forceinline__ void count_landed_head(int* count) {
  asm volatile("fence.proxy.async;" ::: "memory");
  asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(count)
               : "memory");
}

// Whether the head's count has reached `need`; the accumulator reads that
// follow see every counted addition.
__device__ __forceinline__ int head_landed(const int* count, int need) {
  int got;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
               : "=r"(got)
               : "l"(count)
               : "memory");
  return got == need;
}

// One of the head's `blocks` landing CTAs has read its accumulator. The last
// returns both of the head's counts to zero; the last head to finish flips the
// accumulator parity in sched[2], so the next backward adds into the
// accumulator this launch's backward zeroed, and returns the head count in
// sched[3] to zero.
__device__ __forceinline__ void count_read_block(int* reads, int* count,
                                                 int blocks, int* sched,
                                                 int heads) {
  if (atomicAdd(reads, 1) != blocks - 1) return;
  *count = 0;
  *reads = 0;
  if (atomicAdd(sched + 3, 1) == heads - 1) {
    sched[3] = 0;
    sched[2] ^= 1;
  }
}

}  // namespace tileops
