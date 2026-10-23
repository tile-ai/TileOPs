#pragma once

namespace tileops {

// The warpgroup this thread belongs to, broadcast from lane 0 so the compiler
// can prove it uniform across the warp, as CUTLASS's canonical_warp_group_idx
// does. A branch on threadIdx.x selects the same threads, but ptxas cannot
// prove it uniform, and inside it a WGMMA descriptor stays in ordinary
// registers.
__device__ __forceinline__ int canonical_warp_group_idx() {
  return __shfl_sync(0xffffffffu, static_cast<int>(threadIdx.x) / 128, 0);
}

// A pointer broadcast from lane 0. Under register pressure ptxas may keep a
// shared-memory base in an ordinary register; a WGMMA descriptor built from it
// then needs an R2UR before every issue. The broadcast value is provably
// uniform, so the descriptor stays in uniform registers.
template <typename T>
__device__ __forceinline__ T* warp_uniform_ptr(T* ptr) {
  return reinterpret_cast<T*>(
      __shfl_sync(0xffffffffu, reinterpret_cast<unsigned long long>(ptr), 0));
}

// An int broadcast from lane 0. Every lane of the warp may read the same shared
// word, but ptxas cannot prove the result uniform; everything computed from it
// then sits in ordinary registers and needs an R2UR before it feeds a WGMMA
// descriptor or a barrier address.
__device__ __forceinline__ int warp_uniform_int(int value) {
  return __shfl_sync(0xffffffffu, value, 0);
}

// Marks a kernel whose WGMMA descriptor bases a CUDA post-processing pass
// wraps in warp_uniform_ptr. It emits no code.
__device__ __forceinline__ void uniform_wgmma_bases() {}

}  // namespace tileops
