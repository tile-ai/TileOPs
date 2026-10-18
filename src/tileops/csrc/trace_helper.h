namespace tileops {

__device__ __forceinline__ unsigned long long trace_now() {
  return (unsigned long long)clock64();  // per-SM cycle counter (CUDA builtin)
}

__device__ __forceinline__ int trace_thread_idx_x() {
  return threadIdx.x;  // Writer-election fallback for implicit thread blocks
}

// The launch's flat block index. Defined outside the kernel, it reads the
// hardware blockIdx even where a rasterized kernel shadows blockIdx with a
// remapped tile index.
__device__ __forceinline__ int trace_cta_index() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

}  // namespace tileops
