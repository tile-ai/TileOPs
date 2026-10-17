#pragma once

#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

namespace tileops {

// One 64-wide K step of a 64 x N output tile of a 16-bit NT GEMM, issued as one
// N-wide WGMMA per 16 of K. TileLang 0.1.12 issues a ``T.wgmma_gemm`` of width
// N as gcd(N, 256)-wide instructions, each re-reading A from shared memory: 176
// runs as eleven n16 WGMMAs, 192 as three n64. The accumulator holds what those
// narrower instructions would: a thread's column c sits in register
// 4 * (c / 8) + c % 2 of its upper row and two past that of its lower row,
// whatever the instruction width.
//
// a_smem and b_smem are one stage's 64 x 64 A tile and N x 64 B tile, each in
// the 128-byte swizzle ``make_swizzled_layout`` gives a 64-element 16-bit row.
// accumulate == 0 starts the tile's sum, as ``clear_accum`` does.
template <tl::DataType AB, int N, typename T>
__device__ __forceinline__ void dense_wgmma_64xN_k64(float* accumulator,
                                                     T* a_smem, T* b_smem,
                                                     int accumulate) {
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, a_smem);
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, b_smem);
  tl::warpgroup_fence_operand(accumulator, N / 2);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    tl::wgmma_ss<AB, AB, tl::DataType::kFloat32, 64, N, 16, false, false, 1, 1>(
        uint64_t(desc_a + ((ki * 32) >> 4)),
        uint64_t(desc_b + ((ki * 32) >> 4)),
        reinterpret_cast<uint32_t*>(accumulator), ki > 0 || accumulate != 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(accumulator, N / 2);
}

// The instantiations the coop2 kernel names, one per dtype and width.
__device__ __forceinline__ void dense_wgmma_bf16_64x176_k64(float* accumulator,
                                                            bfloat16_t* a_smem,
                                                            bfloat16_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kBFloat16, 176>(accumulator, a_smem,
                                                     b_smem, accumulate);
}

__device__ __forceinline__ void dense_wgmma_bf16_64x192_k64(float* accumulator,
                                                            bfloat16_t* a_smem,
                                                            bfloat16_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kBFloat16, 192>(accumulator, a_smem,
                                                     b_smem, accumulate);
}

__device__ __forceinline__ void dense_wgmma_fp16_64x176_k64(float* accumulator,
                                                            half_t* a_smem,
                                                            half_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kFloat16, 176>(accumulator, a_smem, b_smem,
                                                    accumulate);
}

__device__ __forceinline__ void dense_wgmma_fp16_64x192_k64(float* accumulator,
                                                            half_t* a_smem,
                                                            half_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kFloat16, 192>(accumulator, a_smem, b_smem,
                                                    accumulate);
}

// One 128-wide K step of a 64 x N output tile of an FP8 (e4m3) NT GEMM, issued
// as one N-wide WGMMA per 32 of K, with the accumulator layout and the 128-byte
// swizzled stage of ``dense_wgmma_64xN_k64``: a 128-element fp8 row is 128
// bytes.
template <int N>
__device__ __forceinline__ void dense_wgmma_fp8_64xN_k128(float* accumulator,
                                                          fp8_e4_t* a_smem,
                                                          fp8_e4_t* b_smem,
                                                          int accumulate) {
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, a_smem);
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, b_smem);
  tl::warpgroup_fence_operand(accumulator, N / 2);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    tl::wgmma_ss<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3,
                 tl::DataType::kFloat32, 64, N, 32, false, false, 1, 1>(
        uint64_t(desc_a + ((ki * 32) >> 4)),
        uint64_t(desc_b + ((ki * 32) >> 4)),
        reinterpret_cast<uint32_t*>(accumulator), ki > 0 || accumulate != 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(accumulator, N / 2);
}

// The instantiation the per-tensor FP8 warp-specialized kernel names.
__device__ __forceinline__ void dense_wgmma_fp8_64x176_k128(float* accumulator,
                                                            fp8_e4_t* a_smem,
                                                            fp8_e4_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_fp8_64xN_k128<176>(accumulator, a_smem, b_smem, accumulate);
}

}  // namespace tileops
