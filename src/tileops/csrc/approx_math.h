#pragma once

namespace tl {

// One MUFU.RCP. ``__fdividef`` reaches the same instruction for a normal
// divisor, but its PTX is ``div.approx.f32``, which ptxas guards with a test, a
// scale and two selects so that a subnormal divisor still returns a subnormal
// result.
__device__ __forceinline__ float approx_reciprocal(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// One MUFU.EX2, which flushes a subnormal result to zero. ``exp2f`` guards the
// same instruction with a test and two multiplies to keep it.
__device__ __forceinline__ float approx_exp2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

}  // namespace tl
