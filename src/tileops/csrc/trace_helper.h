__device__ __forceinline__ unsigned long long __tl_now() {
  return (unsigned long long)clock64();  // per-SM cycle counter (CUDA builtin)
}

__device__ __forceinline__ int __tl_thread_idx_x() {
  return threadIdx.x;  // Writer-election fallback for implicit thread blocks
}
