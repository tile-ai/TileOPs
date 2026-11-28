#pragma once

namespace tileops {

// x rounded half to even and saturated to [-128, 127], a NaN to 0: one
// cvt.rni.sat.s8.f32, the conversion vLLM's int8 quantization uses.
__device__ __forceinline__ signed char round_sat_s8(float x) {
  int code;
  asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(code) : "f"(x));
  return static_cast<signed char>(code);
}

}  // namespace tileops
