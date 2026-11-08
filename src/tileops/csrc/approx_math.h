#pragma once

namespace tileops {

// One MUFU.EX2, which flushes a subnormal result to zero. ``exp2f`` guards the
// same instruction with a test and two multiplies to keep it.
__device__ __forceinline__ float approx_exp2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// One MUFU.TANH, max relative error 2^-10.987. ``tanhf`` is a software sequence
// accurate to a few ulp.
__device__ __forceinline__ float approx_tanh(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// One MUFU.RCP. Not ``asm volatile`` as TileLang's ``fast_rcp`` is, so nvcc
// hoists a loop-invariant reciprocal and merges repeated ones.
__device__ __forceinline__ float approx_rcp(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

}  // namespace tileops
