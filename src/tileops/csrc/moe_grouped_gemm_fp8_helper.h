#pragma once

// WGMMA, promotion and epilogue helpers of the masked FP8 M-grouped GEMM,
// copied from the dense FP8 1D2D kernel's fp8_1d2d_helper.h (the parts this
// kernel calls), so that header can change without moving this kernel.

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
// order DeepGEMM's promotion uses, so the two agree bit for bit. The scales
// arrive by value: the stage is released, so nothing here may read it.
template <int BlockN>
__device__ __forceinline__ void moe_fp8_promote(float* partial,
                                                float* final_accum,
                                                float scale_a_row0,
                                                float scale_a_row1,
                                                float scale_b) {
  float const scale0 = scale_a_row0 * scale_b;
  float const scale1 = scale_a_row1 * scale_b;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    final_accum[i * 4 + 0] += scale0 * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += scale0 * partial[i * 4 + 1];
    final_accum[i * 4 + 2] += scale1 * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += scale1 * partial[i * 4 + 3];
  }
}

// A constexpr gcd, for the split points a tile width admits.
__host__ __device__ constexpr int moe_fp8_gcd(int a, int b) {
  return b == 0 ? a : moe_fp8_gcd(b, a % b);
}

// The promotion of a tile spanning two ``b`` scale rows: its first
// ``FirstIters`` eight-column fragments take ``scale_b_0``, the rest
// ``scale_b_1``. The four scale products are folded once, before the
// fragment loop, and the split is a constant, so no fragment selects a scale.
template <int BlockN, int FirstIters>
__device__ __forceinline__ void moe_fp8_promote_two_b_scales_split(
    float* partial, float* final_accum, float scale_a_row0, float scale_a_row1,
    float scale_b_0, float scale_b_1) {
  float const scale_0_0 = scale_a_row0 * scale_b_0;
  float const scale_1_0 = scale_a_row1 * scale_b_0;
  float const scale_0_1 = scale_a_row0 * scale_b_1;
  float const scale_1_1 = scale_a_row1 * scale_b_1;
#pragma unroll
  for (int i = 0; i < FirstIters; ++i) {
    final_accum[i * 4 + 0] += scale_0_0 * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += scale_0_0 * partial[i * 4 + 1];
    final_accum[i * 4 + 2] += scale_1_0 * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += scale_1_0 * partial[i * 4 + 3];
  }
#pragma unroll
  for (int i = FirstIters; i < BlockN / 8; ++i) {
    final_accum[i * 4 + 0] += scale_0_1 * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += scale_0_1 * partial[i * 4 + 1];
    final_accum[i * 4 + 2] += scale_1_1 * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += scale_1_1 * partial[i * 4 + 3];
  }
}

// ``first_iters`` is ``(128 - n0 % 128) / 8`` for a tile starting at ``n0``, a
// multiple of BlockN, so it is a multiple of ``gcd(128, BlockN) / 8`` up to 16:
// 4, 8, 12 or 16 for BlockN 160, 8 or 16 for 192. Each value gets its own
// constant-split body; the branch is warp-uniform and taken once per K step.
template <int BlockN, int Split>
__device__ __forceinline__ void moe_fp8_promote_two_b_scales_dispatch(
    float* partial, float* final_accum, float scale_a_row0, float scale_a_row1,
    float scale_b_0, float scale_b_1, int first_iters) {
  constexpr int kGap = moe_fp8_gcd(128, BlockN) / 8;
  if constexpr (Split + kGap <= 128 / 8) {
    if (first_iters == Split) {
      moe_fp8_promote_two_b_scales_split<BlockN, Split>(
          partial, final_accum, scale_a_row0, scale_a_row1, scale_b_0,
          scale_b_1);
      return;
    }
    moe_fp8_promote_two_b_scales_dispatch<BlockN, Split + kGap>(
        partial, final_accum, scale_a_row0, scale_a_row1, scale_b_0, scale_b_1,
        first_iters);
  } else {
    moe_fp8_promote_two_b_scales_split<BlockN, Split>(
        partial, final_accum, scale_a_row0, scale_a_row1, scale_b_0, scale_b_1);
  }
}

template <int BlockN>
__device__ __forceinline__ void moe_fp8_promote_two_b_scales(
    float* partial, float* final_accum, float scale_a_row0, float scale_a_row1,
    float scale_b_0, float scale_b_1, int first_iters) {
  static_assert(128 % BlockN != 0 && BlockN > 128 && BlockN <= 256,
                "two b scale rows span a tile wider than 128 that 128 does "
                "not divide");
  moe_fp8_promote_two_b_scales_dispatch<BlockN, moe_fp8_gcd(128, BlockN) / 8>(
      partial, final_accum, scale_a_row0, scale_a_row1, scale_b_0, scale_b_1,
      first_iters);
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
    nv_bfloat162 v0 =
        __float22bfloat162_rn({accumulator[i * 4 + 0], accumulator[i * 4 + 1]});
    nv_bfloat162 v1 =
        __float22bfloat162_rn({accumulator[i * 4 + 2], accumulator[i * 4 + 3]});
    cute::SM90_U32x2_STSM_N::copy(*reinterpret_cast<uint32_t*>(&v0),
                                  *reinterpret_cast<uint32_t*>(&v1), *dst);
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

// The block_n widths the builder can name: the WGMMA and the staging store of
// every width, and the promotion that reads one or two ``b`` scale rows.
#define TILEOPS_DEFINE_MOE_FP8_HELPERS(N)                                  \
  __device__ __forceinline__ void moe_fp8_wgmma_64x128_by_128x##N##_lo(    \
      float* acc, uint32_t a_lo, uint32_t b_lo) {                          \
    moe_fp8_wgmma_64x128_by_128xN_lo<N>(acc, a_lo, b_lo);                  \
  }                                                                        \
  __device__ __forceinline__ void moe_fp8_stsm_bf16_swizzled_bm64_64x##N(  \
      float* acc, bfloat16_t* out, int m_offset) {                         \
    moe_fp8_stsm_bf16_swizzled<64, N>(acc, out, m_offset);                 \
  }                                                                        \
  __device__ __forceinline__ void moe_fp8_stsm_bf16_swizzled_bm128_64x##N( \
      float* acc, bfloat16_t* out, int m_offset) {                         \
    moe_fp8_stsm_bf16_swizzled<128, N>(acc, out, m_offset);                \
  }

#define TILEOPS_DEFINE_MOE_FP8_ONE_B_SCALE_PROMOTE(N)       \
  __device__ __forceinline__ void moe_fp8_promote_64x##N(   \
      float* p, float* f, float sa0, float sa1, float sb) { \
    moe_fp8_promote<N>(p, f, sa0, sa1, sb);                 \
  }

#define TILEOPS_DEFINE_MOE_FP8_TWO_B_SCALE_PROMOTE(N)                       \
  __device__ __forceinline__ void moe_fp8_promote_two_b_scales_64x##N(      \
      float* p, float* f, float sa0, float sa1, float sb0, float sb1,       \
      int first_iters) {                                                    \
    moe_fp8_promote_two_b_scales<N>(p, f, sa0, sa1, sb0, sb1, first_iters); \
  }

TILEOPS_DEFINE_MOE_FP8_HELPERS(64)
TILEOPS_DEFINE_MOE_FP8_HELPERS(128)
TILEOPS_DEFINE_MOE_FP8_HELPERS(160)
TILEOPS_DEFINE_MOE_FP8_HELPERS(192)
TILEOPS_DEFINE_MOE_FP8_ONE_B_SCALE_PROMOTE(64)
TILEOPS_DEFINE_MOE_FP8_ONE_B_SCALE_PROMOTE(128)
TILEOPS_DEFINE_MOE_FP8_TWO_B_SCALE_PROMOTE(160)
TILEOPS_DEFINE_MOE_FP8_TWO_B_SCALE_PROMOTE(192)

#undef TILEOPS_DEFINE_MOE_FP8_HELPERS
#undef TILEOPS_DEFINE_MOE_FP8_ONE_B_SCALE_PROMOTE
#undef TILEOPS_DEFINE_MOE_FP8_TWO_B_SCALE_PROMOTE

}  // namespace tileops
