#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

#include <cstdint>

namespace tileops {

// Two e4m3 codes, the low byte first, as a bfloat16 pair: the code widens
// exactly to half and to float, and every e4m3 value is a bfloat16 value.
__device__ __forceinline__ uint32_t w8a16_pair(uint32_t codes) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(codes), __NV_E4M3);
  float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
  __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<uint32_t*>(&b);
}

// One 128-wide K step of a 64 x N tile, weights times activations, into a fresh
// float32 accumulator: the warpgroup's 64 rows of a [rows][128] e4m3 tile
// (`codes`, its first row) widened to bfloat16 in registers as WGMMA's A
// operand, and the [2][N][64] bfloat16 activation tile, its two 64-column
// halves one after the other, as its B operand. Both tiles lie in the 128-byte
// swizzle (16-byte chunk c of row r at chunk c ^ (r % 8)) at 1024-byte-aligned
// addresses. A thread's A registers for K step kk hold rows g and g + 8,
// columns 2q, 2q + 1 and 2q + 8, 2q + 9 of chunk kk, g = 16 * warp + lane / 4,
// q = lane % 4.
template <int N>
__device__ __forceinline__ void w8a16_wgmma_step(float* accumulator,
                                                 const void* codes,
                                                 bfloat16_t* act) {
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x >> 5) & 3;
  const uint8_t* base = static_cast<const uint8_t*>(codes) + (lane & 3) * 2;
  uint32_t a[32];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = warp * 16 + (lane >> 2) + h * 8;
      const uint8_t* chunk = base + row * 128 + ((kk ^ (row & 7)) << 4);
      a[4 * kk + h] = w8a16_pair(*reinterpret_cast<const uint16_t*>(chunk));
      a[4 * kk + 2 + h] =
          w8a16_pair(*reinterpret_cast<const uint16_t*>(chunk + 8));
    }
  }
  tl::GmmaDescriptor desc;
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc, act);
  tl::warpgroup_fence_operand(accumulator, N / 2);
  tl::warpgroup_arrive();
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                 tl::DataType::kFloat32, 64, N, 16, false, false>(
        a + 4 * kk,
        uint64_t(desc + (((kk >> 2) * N * 128 + (kk & 3) * 32) >> 4)),
        reinterpret_cast<uint32_t*>(accumulator), kk > 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(accumulator, N / 2);
}

}  // namespace tileops
