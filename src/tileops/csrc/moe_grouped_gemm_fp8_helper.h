#pragma once

// WGMMA, promotion and epilogue helpers of the masked FP8 M-grouped GEMM. The
// WGMMA and epilogue helpers are copied from the dense FP8 1D2D kernel's
// fp8_1d2d_helper.h, so that header can change without moving this kernel; the
// promotion here also serves tiles whose columns span several ``b`` scale rows.

#include <cuda.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

#include <cute/arch/copy_sm90.hpp>

namespace tileops {

// The low 32 bits of a 128B-swizzled K-major WGMMA descriptor. Broadcast from
// lane 0 so NVCC keeps this and every offset added to it uniform.
__device__ __forceinline__ uint32_t moe_fp8_wgmma_desc_lo(fp8_e4_t* smem) {
  tl::GmmaDescriptor desc;
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc, smem);
  return __shfl_sync(0xffffffff, desc.reg32_[0], 0);
}

// One 64 x BlockN x 128 product of a warp-group, overwriting ``accumulator``.
// ``a_lo`` and ``b_lo`` are descriptor low words; the high word is constant.
template <int BlockN>
__device__ __forceinline__ void moe_fp8_wgmma_64x128_by_128xN_lo(
    float* accumulator, uint32_t a_lo, uint32_t b_lo) {
  constexpr uint64_t kDescHi = uint64_t(0x40000040u) << 32;
  tl::warpgroup_fence_operand(accumulator, BlockN / 2);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    tl::wgmma_ss<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3,
                 tl::DataType::kFloat32, 64, BlockN, 32, false, false, 1, 1>(
        kDescHi | uint64_t(a_lo + ki * 2), kDescHi | uint64_t(b_lo + ki * 2),
        reinterpret_cast<uint32_t*>(accumulator), 0 < ki ? 1 : 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(accumulator, BlockN / 2);
}

// final += partial * scale_a * scale_b, the two scales multiplied first, the
// order DeepGEMM's promotion uses, so the two agree bit for bit. The tile's
// columns run in sizeof...(scale_b) sub-blocks of equal width, each inside one
// ``b`` scale row, and ``scale_b`` holds one scale per sub-block. Eight-column
// fragment ``i`` lies in sub-block ``i / (sub-block width / 8)``, a constant,
// so no fragment selects a scale. The scales arrive by value: the stage is
// released, so nothing here may read it.
template <int BlockN, typename... ScaleB>
__device__ __forceinline__ void moe_fp8_promote(float* partial,
                                                float* final_accum,
                                                float scale_a_row0,
                                                float scale_a_row1,
                                                ScaleB... scale_b) {
  constexpr int kSubBlocks = sizeof...(ScaleB);
  static_assert(kSubBlocks > 0 && BlockN % kSubBlocks == 0,
                "the sub-blocks split the tile evenly");
  constexpr int kSubBlockN = BlockN / kSubBlocks;
  static_assert(kSubBlockN % 8 == 0 && 128 % kSubBlockN == 0,
                "a sub-block is whole fragments and lies inside one 128-column "
                "b scale row at any multiple of its width");
  constexpr int kFragmentsPerSubBlock = kSubBlockN / 8;
  float const scale_b_values[kSubBlocks] = {static_cast<float>(scale_b)...};
  float scale0[kSubBlocks];
  float scale1[kSubBlocks];
#pragma unroll
  for (int j = 0; j < kSubBlocks; ++j) {
    scale0[j] = scale_a_row0 * scale_b_values[j];
    scale1[j] = scale_a_row1 * scale_b_values[j];
  }
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    int const j = i / kFragmentsPerSubBlock;
    final_accum[i * 4 + 0] += scale0[j] * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += scale0[j] * partial[i * 4 + 1];
    final_accum[i * 4 + 2] += scale1[j] * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += scale1[j] * partial[i * 4 + 3];
  }
}

// The warp-group's 64 x BlockN accumulator, in bfloat16, into the swizzled
// [BlockN / W][BlockM][W] staging tile a TMA store reads, rows from
// ``m_offset``; W is 64 columns (128B swizzle) when BlockN * 2 bytes is a
// multiple of 128, else 32 (64B swizzle). The builder's descriptor box and
// swizzle follow the same rule.
template <int BlockM, int BlockN>
__device__ __forceinline__ void moe_fp8_stsm_bf16_swizzled(
    float* accumulator, bfloat16_t* output_smem, int m_offset) {
  constexpr int kElemBytes = sizeof(bfloat16_t);
  constexpr int kTileBytes = BlockN * kElemBytes;
  static_assert(kTileBytes % 64 == 0,
                "a staged row is whole 64B swizzle atoms");
  constexpr int kSwizzleBytes = kTileBytes % 128 == 0 ? 128 : 64;
  constexpr int kTmaBlockN = kSwizzleBytes / kElemBytes;
  constexpr int kBankGroups = kSwizzleBytes / 16;
  constexpr int kWgmmaMPerWarp = 16;
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const warp_in_group = (static_cast<int>(threadIdx.x) >> 5) & 3;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    int const atom_offset = i / (kTmaBlockN / 8);
    int const in_atom_offset = i % (kTmaBlockN / 8);
    int const bank_group_index = in_atom_offset + lane * kBankGroups;
    int const row =
        kBankGroups == 8 ? in_atom_offset / 8 + lane : bank_group_index / 8;
    int col = kBankGroups == 8 ? in_atom_offset : bank_group_index % 8;
    col ^= row % kBankGroups;
    auto* dst = reinterpret_cast<cute::uint128_t*>(
        reinterpret_cast<uint8_t*>(output_smem) +
        warp_in_group * (kWgmmaMPerWarp * kSwizzleBytes) +
        m_offset * kSwizzleBytes + atom_offset * BlockM * kSwizzleBytes +
        row * 128 + col * 16);
    __nv_bfloat162_raw const v0 =
        __float22bfloat162_rn({accumulator[i * 4 + 0], accumulator[i * 4 + 1]});
    __nv_bfloat162_raw const v1 =
        __float22bfloat162_rn({accumulator[i * 4 + 2], accumulator[i * 4 + 3]});
    cute::SM90_U32x2_STSM_N::copy((uint32_t(v0.x) | (uint32_t(v0.y) << 16)),
                                  (uint32_t(v1.x) | (uint32_t(v1.y) << 16)),
                                  *dst);
  }
}

// One TMA store of a staged box to a rank-3 tensor at {x, y, z}; the box is
// clipped at the tensor's extents. The caller commits the bulk group and waits
// for it, so the wait can be deferred.
TL_DEVICE void moe_fp8_tma_store_3d_issue(const CUtensorMap& descriptor,
                                          void const* smem_ptr, int x, int y,
                                          int z) {
  uint64_t desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t src = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
      "[%0, {%2, %3, %4}], [%1];"
      :
      : "l"(desc), "r"(src), "r"(x), "r"(y), "r"(z)
      : "memory");
}

// The block_n widths the builder can name: the WGMMA, the promotion and the
// staging store of every width.
#define TILEOPS_DEFINE_MOE_FP8_HELPERS(N)                                  \
  __device__ __forceinline__ void moe_fp8_wgmma_64x128_by_128x##N##_lo(    \
      float* acc, uint32_t a_lo, uint32_t b_lo) {                          \
    moe_fp8_wgmma_64x128_by_128xN_lo<N>(acc, a_lo, b_lo);                  \
  }                                                                        \
  template <typename... ScaleB>                                            \
  __device__ __forceinline__ void moe_fp8_promote_64x##N(                  \
      float* p, float* f, float sa0, float sa1, ScaleB... sb) {            \
    moe_fp8_promote<N>(p, f, sa0, sa1, sb...);                             \
  }                                                                        \
  __device__ __forceinline__ void moe_fp8_stsm_bf16_swizzled_bm64_64x##N(  \
      float* acc, bfloat16_t* out, int m_offset) {                         \
    moe_fp8_stsm_bf16_swizzled<64, N>(acc, out, m_offset);                 \
  }                                                                        \
  __device__ __forceinline__ void moe_fp8_stsm_bf16_swizzled_bm128_64x##N( \
      float* acc, bfloat16_t* out, int m_offset) {                         \
    moe_fp8_stsm_bf16_swizzled<128, N>(acc, out, m_offset);                \
  }

TILEOPS_DEFINE_MOE_FP8_HELPERS(64)
TILEOPS_DEFINE_MOE_FP8_HELPERS(128)
TILEOPS_DEFINE_MOE_FP8_HELPERS(160)
TILEOPS_DEFINE_MOE_FP8_HELPERS(192)

#undef TILEOPS_DEFINE_MOE_FP8_HELPERS

}  // namespace tileops
