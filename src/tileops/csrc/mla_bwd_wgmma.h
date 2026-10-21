#pragma once

#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/ldsm.h>

#include "warp_uniform.h"

namespace tileops {

// The 48-key WGMMAs of the MLA backward, one instruction per 16 of K. TileLang
// 0.1.12 issues a ``T.wgmma_gemm`` of width 48 as three n16 WGMMAs (gcd(48,
// 256)), each re-reading A; these issue n48. The accumulator registers are the
// ones the three n16 instructions would write: a thread's column c sits in
// register 4 * (c / 8) + c % 2 of its upper row and two past that of its lower
// row.
//
// Shared layouts, as ``make_swizzled_layout`` and
// ``make_quarter_bank_swizzled_layout`` lay them out:
//  - a row-major ``R x D`` 16-bit tile with 128-byte swizzle stores each
//    64-column atom as its own ``R x 64`` block of ``R * 128`` bytes;
//  - a row-major ``64 x 48`` 16-bit tile with 32-byte swizzle stores each
//    16-column atom as its own ``64 x 16`` block of 2048 bytes.

// acc[64 x 48] (+)= A[64 x K] * B[48 x K]^T; A and B K-major with 128-byte
// swizzle. a_rows and b_rows are the row counts of the tiles, which set the
// atom stride.
template <tl::DataType AB, int K, int A_ROWS, int B_ROWS, typename T>
__device__ __forceinline__ void mla_wgmma_nt_x48(float* accumulator, T* a_smem,
                                                 T* b_smem, int accumulate) {
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, warp_uniform_ptr(a_smem));
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, warp_uniform_ptr(b_smem));
  tl::warpgroup_fence_operand(accumulator, 24);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < K / 16; ++ki) {
    int a_off = (ki >> 2) * A_ROWS * 128 + (ki & 3) * 32;
    int b_off = (ki >> 2) * B_ROWS * 128 + (ki & 3) * 32;
    tl::wgmma_ss<AB, AB, tl::DataType::kFloat32, 64, 48, 16, false, false, 1,
                 1>(
        uint64_t(desc_a + (a_off >> 4)), uint64_t(desc_b + (b_off >> 4)),
        reinterpret_cast<uint32_t*>(accumulator), ki > 0 || accumulate != 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(accumulator, 24);
}

// acc[M x 48] += A[64 x M]^T * B[64 x 48]; A is a row-major 64-row tile with
// 128-byte swizzle read M-major, B a 64 x 48 tile with 32-byte swizzle read
// N-major. M is a multiple of 64: one M atom per 64 columns of A, 24
// accumulator registers each.
template <tl::DataType AB, int M, typename T>
__device__ __forceinline__ void mla_wgmma_tn_x48(float* accumulator, T* a_smem,
                                                 T* b_smem) {
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  tl::initialize_wgmma_descriptor<1, 512, 64>(desc_a, warp_uniform_ptr(a_smem));
  tl::initialize_wgmma_descriptor<3, 128, 16>(desc_b, warp_uniform_ptr(b_smem));
  tl::warpgroup_fence_operand(accumulator, M / 64 * 24);
  tl::warpgroup_arrive();
#pragma unroll
  for (int mi = 0; mi < M / 64; ++mi) {
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      tl::wgmma_ss<AB, AB, tl::DataType::kFloat32, 64, 48, 16, true, true, 1,
                   1>(uint64_t(desc_a + ((mi * 8192 + ki * 2048) >> 4)),
                      uint64_t(desc_b + ((ki * 512) >> 4)),
                      reinterpret_cast<uint32_t*>(accumulator + mi * 24), 1);
    }
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(accumulator, M / 64 * 24);
}

#define TILEOPS_MLA_BWD_WGMMA(TAG, CT, DT)                                     \
  __device__ __forceinline__ void mla_bwd_s_##TAG(float* acc, CT* q, CT* k,    \
                                                  int accumulate) {            \
    mla_wgmma_nt_x48<DT, 192, 64, 48>(acc, q, k, accumulate);                  \
  }                                                                            \
  __device__ __forceinline__ void mla_bwd_dp_##TAG(float* acc, CT* d_o, CT* v, \
                                                   int accumulate) {           \
    mla_wgmma_nt_x48<DT, 128, 64, 48>(acc, d_o, v, accumulate);                \
  }                                                                            \
  __device__ __forceinline__ void mla_bwd_dvt_##TAG(float* acc, CT* d_o,       \
                                                    CT* p) {                   \
    mla_wgmma_tn_x48<DT, 128>(acc, d_o, p);                                    \
  }                                                                            \
  __device__ __forceinline__ void mla_bwd_dkt_##TAG(float* acc, CT* q,         \
                                                    CT* ds) {                  \
    mla_wgmma_tn_x48<DT, 192>(acc, q, ds);                                     \
  }

TILEOPS_MLA_BWD_WGMMA(bf16, bfloat16_t, tl::DataType::kBFloat16)
TILEOPS_MLA_BWD_WGMMA(fp16, half_t, tl::DataType::kFloat16)

#undef TILEOPS_MLA_BWD_WGMMA

}  // namespace tileops

namespace tileops {

__device__ __forceinline__ uint32_t mla_bwd_pack(float lo, float hi,
                                                 bfloat16_t*) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t mla_bwd_pack(float lo, float hi, half_t*) {
  __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Store a warpgroup's 64 x 48 f32 accumulator (the n48 WGMMA layout) as 16-bit
// values into a 64 x 48 tile with 32-byte swizzle, one stmatrix.x4 per 16 x 16
// block: the accumulator's register quads are the four 8 x 8 matrices of a
// block. A row is 32 bytes of its 16-column atom, atoms 2048 bytes apart, and
// the 16-byte half of a row a column lands in is flipped on rows 4..7 of
// every 8. t is the thread's index in its warpgroup.
template <typename T>
__device__ __forceinline__ void mla_bwd_stsm_64x48(const float* accumulator,
                                                   T* tile, int t) {
  int lane = t & 31;
  int m = lane >> 3;
  int row = (t >> 5) * 16 + (m & 1) * 8 + (lane & 7);
  int half = (m >> 1) ^ ((row >> 2) & 1);
  char* base = reinterpret_cast<char*>(tile) + row * 32 + half * 16;
#pragma unroll
  for (int b = 0; b < 3; ++b) {
    const float* a = accumulator + 8 * b;
    tl::ptx_stmatrix_m8n8_x4(base + b * 2048, mla_bwd_pack(a[0], a[1], tile),
                             mla_bwd_pack(a[2], a[3], tile),
                             mla_bwd_pack(a[4], a[5], tile),
                             mla_bwd_pack(a[6], a[7], tile));
  }
}

__device__ __forceinline__ void mla_bwd_stsm_bf16(const float* accumulator,
                                                  bfloat16_t* tile, int t) {
  mla_bwd_stsm_64x48(accumulator, tile, t);
}

__device__ __forceinline__ void mla_bwd_stsm_fp16(const float* accumulator,
                                                  half_t* tile, int t) {
  mla_bwd_stsm_64x48(accumulator, tile, t);
}

}  // namespace tileops
