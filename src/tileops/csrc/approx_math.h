#pragma once

namespace tileops {

// One MUFU.EX2, which flushes a subnormal result to zero. ``exp2f`` guards the
// same instruction with a test and two multiplies to keep it.
__device__ __forceinline__ float approx_exp2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

}  // namespace tileops
