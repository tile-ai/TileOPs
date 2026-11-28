#pragma once

namespace tileops {

// Meet the `parts` blocks that share the counter at `arrived`, as cooperative
// groups' grid barrier meets a whole grid: every block adds 1, except the
// `leader`, which adds 2^31 - (parts - 1), so the counter's top bit flips once
// all have arrived and its low bits return to where they were. A counter left
// with zero low bits therefore serves every later barrier and launch. Every
// one of the blocks must be resident, as a cooperative launch keeps them.
__device__ __forceinline__ void row_barrier(void* arrived, int parts,
                                            bool leader) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned* counter = reinterpret_cast<unsigned*>(arrived);
    unsigned add = leader ? 0x80000000u - static_cast<unsigned>(parts - 1) : 1u;
    unsigned before;
    asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;"
                 : "=r"(before)
                 : "l"(counter), "r"(add)
                 : "memory");
    unsigned now;
    do {
      asm volatile("ld.acquire.gpu.u32 %0, [%1];"
                   : "=r"(now)
                   : "l"(counter)
                   : "memory");
    } while (((before ^ now) & 0x80000000u) == 0);
  }
  __syncthreads();
}

}  // namespace tileops
