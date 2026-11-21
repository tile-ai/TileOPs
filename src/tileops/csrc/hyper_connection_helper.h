#pragma once

#include <cuda_bf16.h>

#include <cstdint>

#include "bf16_widen.h"

namespace tileops {

// Byte offset of 16-byte chunk ``chunk`` of row ``row`` in a tile of 64-element
// 16-bit rows in the 128-byte swizzle ``make_swizzled_layout`` gives it.
__device__ __forceinline__ int hc_swizzled(int row, int chunk) {
  return row * 128 + ((chunk ^ (row & 7)) << 4);
}

// Splits a Q x 64 float32 tile in shared memory (256-byte rows, unswizzled)
// into a 3Q x 64 tile of its bfloat16 terms at ``terms`` (swizzled as above),
// term t of row r in row t * Q + r. Thread ``tid`` of ``NT`` takes the 4-value
// units tid, tid + NT, ... of the tile's 16 a row. A term is the upper half of
// the float32 left after the terms before it: truncating keeps each remainder
// exact, and the last has at most eight significant bits, so above float32's
// subnormal range the terms sum to the value.
template <int Q, int NT>
__device__ __forceinline__ void hc_split_tile(const float* tile, void* terms,
                                              int tid) {
  static_assert(Q * 16 % NT == 0, "every thread splits as many units");
  constexpr int kUnits = Q * 16 / NT;
  char* base = reinterpret_cast<char*>(terms);
  // Every unit is read before any term is stored, so no load waits on a store.
  float4 v[kUnits];
#pragma unroll
  for (int j = 0; j < kUnits; ++j) {
    v[j] = reinterpret_cast<const float4*>(tile)[tid + j * NT];
  }
#pragma unroll
  for (int j = 0; j < kUnits; ++j) {
    const int u = tid + j * NT;
    const float in[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
    uint32_t w[3][4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      w[0][e] = __float_as_uint(in[e]) & 0xffff0000u;
      const float rest = in[e] - __uint_as_float(w[0][e]);
      w[1][e] = __float_as_uint(rest) & 0xffff0000u;
      w[2][e] = __float_as_uint(rest - __uint_as_float(w[1][e]));
    }
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      *reinterpret_cast<uint2*>(base + hc_swizzled(t * Q + u / 16, u % 16 / 2) +
                                u % 2 * 8) =
          make_uint2(__byte_perm(w[t][0], w[t][1], 0x7632),
                     __byte_perm(w[t][2], w[t][3], 0x7632));
    }
  }
}

// Adds the sums of the squares of the bfloat16 pairs ``dense_load_a_64x64``
// loaded into ``rows[0]`` (the thread's upper row) and ``rows[1]`` (8 rows
// below). Each call's 16 squares a row are summed apart before they join the
// running sum, so its rounding grows with the steps rather than the values.
__device__ __forceinline__ void hc_square_regs(const uint32_t* a, float* rows) {
  float step[2] = {0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float lo = widen_bf16_lo(&a[i]);
    const float hi = widen_bf16_hi(&a[i]);
    step[i & 1] = fmaf(lo, lo, step[i & 1]);
    step[i & 1] = fmaf(hi, hi, step[i & 1]);
  }
  rows[0] += step[0];
  rows[1] += step[1];
}

// ``value`` summed over the four lanes that differ in bits 0 and 1.
__device__ __forceinline__ float hc_quad_sum(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value + __shfl_xor_sync(0xffffffffu, value, 2);
}

// The sum of ``value`` over the four lanes that differ from this one in the
// lane bits ``Bits`` and ``2 * Bits``: one round of three shuffles, added in
// one order on every lane of the four.
template <int Bits>
__device__ __forceinline__ float hc_sum4(float value) {
  const float a = __shfl_xor_sync(0xffffffffu, value, Bits);
  const float b = __shfl_xor_sync(0xffffffffu, value, 2 * Bits);
  const float c = __shfl_xor_sync(0xffffffffu, value, 3 * Bits);
  return (value + a) + (b + c);
}

// ``value`` times the approximate reciprocal of ``divisor``: within two units
// in the last place of the quotient, where a division is within half of one.
__device__ __forceinline__ float hc_scale_by_rcp(float value, float divisor) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(divisor));
  return value * r;
}

// Sinkhorn on a 4 x 4 matrix held one entry a lane, the lane whose bits 3..0
// are ``4 * i + j`` holding entry (i, j): a row softmax plus ``eps``, a column
// normalization, then ``repeat - 1`` row-then-column normalizations, each
// dividing by the sum plus ``eps``. A warp holds the matrix twice. Each step
// waits on one round of shuffles and one reciprocal, the latency the repeats
// chain.
__device__ __forceinline__ float hc_sinkhorn4(float logit, int repeat,
                                              float eps) {
  const float m = fmaxf(fmaxf(logit, __shfl_xor_sync(0xffffffffu, logit, 1)),
                        fmaxf(__shfl_xor_sync(0xffffffffu, logit, 2),
                              __shfl_xor_sync(0xffffffffu, logit, 3)));
  const float e = expf(logit - m);
  float v = hc_scale_by_rcp(e, hc_sum4<1>(e)) + eps;
  v = hc_scale_by_rcp(v, hc_sum4<4>(v) + eps);
  for (int it = 1; it < repeat; ++it) {
    v = hc_scale_by_rcp(v, hc_sum4<1>(v) + eps);
    v = hc_scale_by_rcp(v, hc_sum4<4>(v) + eps);
  }
  return v;
}

}  // namespace tileops
