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

// The instantiations the coop2 kernel and the persistent template name, one per
// dtype and width.
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

__device__ __forceinline__ void dense_wgmma_bf16_64x144_k64(float* accumulator,
                                                            bfloat16_t* a_smem,
                                                            bfloat16_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kBFloat16, 144>(accumulator, a_smem,
                                                     b_smem, accumulate);
}

__device__ __forceinline__ void dense_wgmma_fp16_64x144_k64(float* accumulator,
                                                            half_t* a_smem,
                                                            half_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kFloat16, 144>(accumulator, a_smem, b_smem,
                                                    accumulate);
}

__device__ __forceinline__ void dense_wgmma_bf16_64x160_k64(float* accumulator,
                                                            bfloat16_t* a_smem,
                                                            bfloat16_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kBFloat16, 160>(accumulator, a_smem,
                                                     b_smem, accumulate);
}

__device__ __forceinline__ void dense_wgmma_fp16_64x160_k64(float* accumulator,
                                                            half_t* a_smem,
                                                            half_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kFloat16, 160>(accumulator, a_smem, b_smem,
                                                    accumulate);
}

__device__ __forceinline__ void dense_wgmma_bf16_64x240_k64(float* accumulator,
                                                            bfloat16_t* a_smem,
                                                            bfloat16_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kBFloat16, 240>(accumulator, a_smem,
                                                     b_smem, accumulate);
}

__device__ __forceinline__ void dense_wgmma_fp16_64x240_k64(float* accumulator,
                                                            half_t* a_smem,
                                                            half_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kFloat16, 240>(accumulator, a_smem, b_smem,
                                                    accumulate);
}

__device__ __forceinline__ void dense_wgmma_bf16_64x256_k64(float* accumulator,
                                                            bfloat16_t* a_smem,
                                                            bfloat16_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kBFloat16, 256>(accumulator, a_smem,
                                                     b_smem, accumulate);
}

__device__ __forceinline__ void dense_wgmma_fp16_64x256_k64(float* accumulator,
                                                            half_t* a_smem,
                                                            half_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kFloat16, 256>(accumulator, a_smem, b_smem,
                                                    accumulate);
}

__device__ __forceinline__ void dense_wgmma_bf16_64x224_k64(float* accumulator,
                                                            bfloat16_t* a_smem,
                                                            bfloat16_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kBFloat16, 224>(accumulator, a_smem,
                                                     b_smem, accumulate);
}

__device__ __forceinline__ void dense_wgmma_fp16_64x224_k64(float* accumulator,
                                                            half_t* a_smem,
                                                            half_t* b_smem,
                                                            int accumulate) {
  dense_wgmma_64xN_k64<tl::DataType::kFloat16, 224>(accumulator, a_smem, b_smem,
                                                    accumulate);
}

// One 64-wide K step of a 128 x N output tile, issued as one N-wide WGMMA per
// 16 of K for each 64-row half (lo, hi), so a width that does not divide 256
// reads each A row once. A_MN / B_MN say whether that operand's stage is
// MN-major (K outer) rather than K-major; the transpose flags follow.
//
// The stages are in the swizzle ``make_swizzled_layout`` gives them. K-major:
// 64-element rows in the 128-byte swizzle. MN-major A: a 64 x 128 stage of two
// 64-column atoms in the 128-byte swizzle. MN-major B: a 64 x BW stage of
// 64-column atoms in the 128-byte swizzle, or of 32-column atoms in the 64-byte
// swizzle when BW is not a multiple of 64; the WGMMA reads its first N columns.
// Each half's accumulator is laid out as ``dense_wgmma_64xN_k64``'s.
template <tl::DataType AB, int N, bool A_MN, bool B_MN, int BW = N, typename T>
__device__ __forceinline__ void dense_wgmma_128xN_k64(float* lo, float* hi,
                                                      T* a_smem, T* b_smem,
                                                      int accumulate) {
  constexpr bool kB64 = B_MN && BW % 64 != 0;
  constexpr int kAStep = A_MN ? 2048 : 32;
  constexpr int kBStep = B_MN ? (kB64 ? 1024 : 2048) : 32;
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  tl::initialize_wgmma_descriptor<1, A_MN ? 0 : 1, 64>(desc_a, a_smem);
  tl::initialize_wgmma_descriptor<kB64 ? 2 : 1, B_MN ? (kB64 ? 256 : 512) : 1,
                                  kB64 ? 32 : 64>(desc_b, b_smem);
  tl::warpgroup_fence_operand(lo, N / 2);
  tl::warpgroup_fence_operand(hi, N / 2);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    tl::wgmma_ss<AB, AB, tl::DataType::kFloat32, 64, N, 16, A_MN, B_MN, 1, 1>(
        uint64_t(desc_a + ((ki * kAStep) >> 4)),
        uint64_t(desc_b + ((ki * kBStep) >> 4)),
        reinterpret_cast<uint32_t*>(lo), ki > 0 || accumulate != 0);
    tl::wgmma_ss<AB, AB, tl::DataType::kFloat32, 64, N, 16, A_MN, B_MN, 1, 1>(
        uint64_t(desc_a + ((8192 + ki * kAStep) >> 4)),
        uint64_t(desc_b + ((ki * kBStep) >> 4)),
        reinterpret_cast<uint32_t*>(hi), ki > 0 || accumulate != 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(lo, N / 2);
  tl::warpgroup_fence_operand(hi, N / 2);
}

// Columns [8 * t0, 8 * t0 + Cols) of one warpgroup's 64-row fp32 WGMMA
// accumulator, cast to T and stored transposed into a Cols x 64 shared tile in
// the 128-byte swizzle: one stmatrix.trans per two 8-column groups and warp.
template <int Cols, typename T>
__device__ __forceinline__ void dense_store_trans_64(const float* acc, T* smem,
                                                     int t0) {
  const int lane = threadIdx.x & 31;
  const int warp = (threadIdx.x >> 5) & 3;
  const uint32_t base = smem_ptr_to_uint(smem);
  auto pack = [](float x, float y) {
    uint32_t r;
    if constexpr (std::is_same_v<T, bfloat16_t>) {
      __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
      r = *reinterpret_cast<uint32_t*>(&v);
    } else {
      __half2 v = __floats2half2_rn(x, y);
      r = *reinterpret_cast<uint32_t*>(&v);
    }
    return r;
  };
#pragma unroll
  for (int g = 0; g < Cols / 8; g += 2) {
    const int row = 8 * (g + (lane >> 4)) + (lane & 7);
    const int unit = (2 * warp + ((lane >> 3) & 1)) ^ (row & 7);
    const float* v = acc + 4 * (t0 + g);
    asm volatile(
        "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, "
        "%4};\n" ::"r"(base + row * 128 + unit * 16),
        "r"(pack(v[0], v[1])), "r"(pack(v[2], v[3])), "r"(pack(v[4], v[5])),
        "r"(pack(v[6], v[7])));
  }
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
