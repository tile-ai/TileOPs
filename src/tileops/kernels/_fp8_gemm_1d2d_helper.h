#pragma once

#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/instruction/wgmma.h>

#include <cuda.h>
#include <cutlass/float8.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <type_traits>

namespace tl {

template <int BlockN>
__device__ __forceinline__ void fp8_gemm_wgmma_64x128_by_128xN(
    float* accumulator, fp8_e4_t* a_smem, fp8_e4_t* b_smem) {
  GmmaDescriptor desc_a;
  GmmaDescriptor desc_b;
  initialize_wgmma_descriptor<1, 1, 64>(desc_a, a_smem);
  initialize_wgmma_descriptor<1, 1, 64>(desc_b, b_smem);
  warpgroup_fence_operand(accumulator, BlockN / 2);
  warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    wgmma_ss<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
             DataType::kFloat32, 64, BlockN, 32, false, false, 1, 1>(
        uint64_t(desc_a + ((ki * 32) >> 4)),
        uint64_t(desc_b + ((ki * 32) >> 4)),
        reinterpret_cast<uint32_t*>(accumulator), 0 < ki ? 1 : 0);
  }
  warpgroup_commit_batch();
  warpgroup_fence_operand(accumulator, BlockN / 2);
}

TL_DEVICE void fp8_tma_store_2d_ptx(const CUtensorMap& descriptor,
                                    void const* smem_ptr, int x, int y) {
  uint64_t desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t src = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
      "[%0, {%2, %3}], [%1];" : : "l"(desc), "r"(src), "r"(x), "r"(y) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// Promote one 64xNx128 WGMMA partial directly in its native per-thread
// accumulator layout.  This deliberately avoids materialising the fragment as
// a logical 2-D TileLang array between every K block.
template <int BlockN>
__device__ __forceinline__ void fp8_gemm_1d2d_promote_shared_ab_uniform(
    float* partial, float* final_accum, const float* scale_a,
    const float* scale_b, int scale_k_idx) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const warp_in_group = (static_cast<int>(threadIdx.x) >> 5) & 3;
  int const row0 = warp_in_group * 16 + lane / 4;
  int const row1 = row0 + 8;
  float const sb = scale_b[scale_k_idx];
  float const scale0 = scale_a[row0] * sb;
  float const scale1 = scale_a[row1] * sb;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    final_accum[i * 4 + 0] += scale0 * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += scale0 * partial[i * 4 + 1];
  }
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    final_accum[i * 4 + 2] += scale1 * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += scale1 * partial[i * 4 + 3];
  }
}

template <int BlockN>
__device__ __forceinline__ void fp8_gemm_raw_acc_stsm_bf16(
    float* accumulator, bfloat16_t* output_smem) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const warp_in_group = (static_cast<int>(threadIdx.x) >> 5) & 3;
#pragma unroll
  for (int i = 0; i < (BlockN / 2) / 8; ++i) {
    nv_bfloat162 v0 = __float22bfloat162_rn(
        {accumulator[i * 8 + 0], accumulator[i * 8 + 1]});
    nv_bfloat162 v1 = __float22bfloat162_rn(
        {accumulator[i * 8 + 2], accumulator[i * 8 + 3]});
    nv_bfloat162 v2 = __float22bfloat162_rn(
        {accumulator[i * 8 + 4], accumulator[i * 8 + 5]});
    nv_bfloat162 v3 = __float22bfloat162_rn(
        {accumulator[i * 8 + 6], accumulator[i * 8 + 7]});
    auto* dst = reinterpret_cast<cute::uint128_t*>(
        reinterpret_cast<__nv_bfloat16*>(output_smem) +
        (warp_in_group * 16 + lane % 16) * BlockN +
        i * 16 + 8 * (lane / 16));
    cute::SM90_U32x4_STSM_N::copy(
        *reinterpret_cast<uint32_t*>(&v0), *reinterpret_cast<uint32_t*>(&v1),
        *reinterpret_cast<uint32_t*>(&v2), *reinterpret_cast<uint32_t*>(&v3), *dst);
  }
}

template <int BlockN, typename OutT>
__device__ __forceinline__ void fp8_gemm_raw_acc_store_global_vec2(
    float* accumulator, OutT* output, int leading_dim, int m_start,
    int n_start, int shape_m, int shape_n) {
  using MutableOutT = std::remove_const_t<OutT>;
  MutableOutT* out = const_cast<MutableOutT*>(output);
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const warp_in_group = (static_cast<int>(threadIdx.x) >> 5) & 3;
  int const row0 = m_start + warp_in_group * 16 + lane / 4;
  int const row1 = row0 + 8;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    int const col = n_start + i * 8 + (lane % 4) * 2;
    if (col + 1 < shape_n) {
      if (row0 < shape_m) {
        uint32_t packed;
        MutableOutT* values = reinterpret_cast<MutableOutT*>(&packed);
        values[0] = static_cast<MutableOutT>(accumulator[i * 4 + 0]);
        values[1] = static_cast<MutableOutT>(accumulator[i * 4 + 1]);
        *reinterpret_cast<uint32_t*>(out + row0 * leading_dim + col) = packed;
      }
      if (row1 < shape_m) {
        uint32_t packed;
        MutableOutT* values = reinterpret_cast<MutableOutT*>(&packed);
        values[0] = static_cast<MutableOutT>(accumulator[i * 4 + 2]);
        values[1] = static_cast<MutableOutT>(accumulator[i * 4 + 3]);
        *reinterpret_cast<uint32_t*>(out + row1 * leading_dim + col) = packed;
      }
    }
  }
}

#define TL_DEFINE_FP8_GEMM_1D2D_HELPERS(N)                                      \
__device__ __forceinline__ void fp8_gemm_wgmma_64x128_by_128x##N(               \
    float* acc, fp8_e4_t* a, fp8_e4_t* b) {                                     \
  fp8_gemm_wgmma_64x128_by_128xN<N>(acc, a, b);                                 \
}                                                                                \
__device__ __forceinline__ void fp8_gemm_raw_acc_stsm_bf16_64x##N(              \
    float* acc, bfloat16_t* out) { fp8_gemm_raw_acc_stsm_bf16<N>(acc, out); }    \
template <typename OutT>                                                         \
__device__ __forceinline__ void fp8_gemm_raw_acc_store_global_64x##N##_v2(       \
    float* acc, OutT* out, int ld, int ms, int ns, int m, int n) {               \
  fp8_gemm_raw_acc_store_global_vec2<N>(acc, out, ld, ms, ns, m, n);             \
}                                                                                \
__device__ __forceinline__ void fp8_gemm_1d2d_promote_shared_ab_uniform_64x##N(  \
    float* p, float* f, const float* sa, const float* sb, int sk) {               \
  fp8_gemm_1d2d_promote_shared_ab_uniform<N>(p, f, sa, sb, sk);                  \
}

TL_DEFINE_FP8_GEMM_1D2D_HELPERS(16)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(32)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(64)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(128)

#undef TL_DEFINE_FP8_GEMM_1D2D_HELPERS

}  // namespace tl
