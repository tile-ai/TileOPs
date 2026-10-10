namespace tileops {

__device__ __forceinline__ unsigned long long trace_now() {
  return (unsigned long long)clock64();  // per-SM cycle counter (CUDA builtin)
}

__device__ __forceinline__ int trace_thread_idx_x() {
  return threadIdx.x;  // Writer-election fallback for implicit thread blocks
}

}  // namespace tileops
