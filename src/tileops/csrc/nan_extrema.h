#pragma once

namespace tileops {

// The larger of a and b, or NaN when either is NaN, as torch.amax propagates
// it: one FMNMX.NAN (sm_80+). fmaxf, which T.max lowers to, returns the other
// operand instead.
__device__ __forceinline__ float max_nan(float a, float b) {
  float r;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// The smaller of a and b, or NaN when either is NaN, as torch.amin propagates
// it: one FMNMX.NAN (sm_80+).
__device__ __forceinline__ float min_nan(float a, float b) {
  float r;
  asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

}  // namespace tileops
