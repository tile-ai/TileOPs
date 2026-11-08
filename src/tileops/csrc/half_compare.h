#pragma once

#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

namespace tileops {

// a != b, true when either is NaN, as torch.ne answers: one HSETP2.NEU.
// cutlass's operator!= is __hne, the ordered compare, which answers false
// against a NaN.
__device__ __forceinline__ bool ne_unordered(cutlass::half_t a,
                                             cutlass::half_t b) {
  return __hneu(a.to_half(), b.to_half());
}

__device__ __forceinline__ bool ne_unordered(cutlass::bfloat16_t a,
                                             cutlass::bfloat16_t b) {
  return __hneu(a.to_nv_bfloat16(), b.to_nv_bfloat16());
}

}  // namespace tileops
