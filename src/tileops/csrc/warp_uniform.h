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

}  // namespace tileops
