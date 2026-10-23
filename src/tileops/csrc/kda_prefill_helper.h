#pragma once

namespace tileops {

// Every thread must take this exit before the chunk's block barriers.
__device__ __forceinline__ void kda_empty_chunk_exit() {
  asm volatile("exit;" ::: "memory");
}

}  // namespace tileops
