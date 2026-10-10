#pragma once

#include <cuda_runtime.h>

#include <cuda/atomic>

namespace tileops {

// Call after a barrier involving every partial writer. The release sequence
// acquires all preceding publications in the last CTA, which then synchronizes
// its readers before reducing the partials.
__device__ __forceinline__ int paged_arrive(int* counter) {
  cuda::atomic_ref<int, cuda::thread_scope_device> value(*counter);
  return value.fetch_add(1, cuda::memory_order_acq_rel);
}

// Keep the intrinsic visible to NVCC so it can schedule independent tanh
// instructions. The PTX fallback also supports CUDA releases before 12.8.
__device__ __forceinline__ float paged_tanh(float x) {
#if (__CUDACC_VER_MAJOR__ > 12) || \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
  return __tanhf(x);
#else
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#endif
}

}  // namespace tileops
