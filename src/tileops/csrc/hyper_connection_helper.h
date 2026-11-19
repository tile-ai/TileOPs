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

// ``lo`` and ``hi`` rounded to bfloat16, packed low half first.
__device__ __forceinline__ uint32_t hc_pack_bf16(float lo, float hi) {
  return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
         (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

// The three bfloat16 terms of the eight float32 values at ``src``, whose sum is
// each value, as eight packed bfloat16 per term.
__device__ __forceinline__ void hc_split8(const float* src, uint4 (&out)[3]) {
  float4 a = reinterpret_cast<const float4*>(src)[0];
  float4 b = reinterpret_cast<const float4*>(src)[1];
  float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  uint32_t w[3][4];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    float r0 = v[2 * p];
    float r1 = v[2 * p + 1];
#pragma unroll
    for (int t = 0; t < 3; ++t) {
      uint32_t packed = hc_pack_bf16(r0, r1);
      w[t][p] = packed;
      r0 -= widen_bf16_lo(&packed);
      r1 -= widen_bf16_hi(&packed);
    }
  }
#pragma unroll
  for (int t = 0; t < 3; ++t)
    out[t] = make_uint4(w[t][0], w[t][1], w[t][2], w[t][3]);
}

// Splits the eight float32 values at ``src`` and stores term t at
// ``dst + t * term_stride`` (bfloat16 elements).
__device__ __forceinline__ void hc_split_global(const float* src, void* dst,
                                                int term_stride) {
  uint4 out[3];
  hc_split8(src, out);
  uint16_t* base = reinterpret_cast<uint16_t*>(dst);
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    *reinterpret_cast<uint4*>(base + t * term_stride) = out[t];
  }
}

// Splits a Q x (64 * Steps) float32 slice of ``fn`` (rows ``ld`` elements
// apart) into Steps tiles of 3Q x 64 bfloat16 terms at ``terms`` (swizzled as
// above), thread ``tid`` of ``NT`` taking every NT-th 8-value chunk, Batch at a
// time.
template <int Q, int Steps, int NT, int Batch>
__device__ __forceinline__ void hc_split_slice(const float* fn, int ld,
                                               void* terms, int tid) {
  constexpr int kChunks = Q * Steps * 8;
  constexpr int kPer = (kChunks + NT - 1) / NT;
  char* base = reinterpret_cast<char*>(terms);
#pragma unroll
  for (int b0 = 0; b0 < kPer; b0 += Batch) {
    float4 v[Batch][2];
#pragma unroll
    for (int u = 0; u < Batch; ++u) {
      int c = tid + (b0 + u) * NT;
      if (b0 + u < kPer && c < kChunks) {
        int step = c / (Q * 8);
        int row = (c / 8) % Q;
        const float4* p = reinterpret_cast<const float4*>(
            fn + row * ld + step * 64 + (c % 8) * 8);
        v[u][0] = __ldg(p);
        v[u][1] = __ldg(p + 1);
      }
    }
#pragma unroll
    for (int u = 0; u < Batch; ++u) {
      int c = tid + (b0 + u) * NT;
      if (b0 + u < kPer && c < kChunks) {
        int step = c / (Q * 8);
        int row = (c / 8) % Q;
        uint4 out[3];
        hc_split8(reinterpret_cast<const float*>(v[u]), out);
        char* tile = base + step * (3 * Q * 128);
#pragma unroll
        for (int t = 0; t < 3; ++t) {
          *reinterpret_cast<uint4*>(tile + hc_swizzled(t * Q + row, c % 8)) =
              out[t];
        }
      }
    }
  }
}

// The sum of squares of half ``tid & 1`` of row ``tid >> 1`` of a 64 x 64
// bfloat16 tile (swizzled as above): 32 values, read as four 16-byte chunks.
__device__ __forceinline__ float hc_square_sum(const void* tile, int tid) {
  const char* base = reinterpret_cast<const char*>(tile);
  int row = tid >> 1;
  int half = tid & 1;
  float s = 0.f;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    uint4 c =
        *reinterpret_cast<const uint4*>(base + hc_swizzled(row, half * 4 + u));
    uint32_t w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float lo = widen_bf16_lo(&w[e]);
      float hi = widen_bf16_hi(&w[e]);
      s = fmaf(lo, lo, s);
      s = fmaf(hi, hi, s);
    }
  }
  return s;
}

// ``value`` plus its value on the lane that differs in bit 0.
__device__ __forceinline__ float hc_pair_sum(float value) {
  return value + __shfl_xor_sync(0xffffffffu, value, 1);
}

}  // namespace tileops
