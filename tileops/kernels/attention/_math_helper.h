#pragma once

namespace tl {

__device__ __forceinline__ float tileops_tanh_approx_f32(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

}  // namespace tl
