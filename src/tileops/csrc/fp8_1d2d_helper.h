#pragma once

#include <cuda.h>
#include <cutlass/float8.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

#include <cstring>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <type_traits>

namespace tileops {

template <int BlockN>
__device__ __forceinline__ void fp8_gemm_wgmma_64x128_by_128xN(
    float* accumulator, fp8_e4_t* a_smem, fp8_e4_t* b_smem) {
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, a_smem);
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, b_smem);
  tl::warpgroup_fence_operand(accumulator, BlockN / 2);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    tl::wgmma_ss<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3,
                 tl::DataType::kFloat32, 64, BlockN, 32, false, false, 1, 1>(
        uint64_t(desc_a + ((ki * 32) >> 4)),
        uint64_t(desc_b + ((ki * 32) >> 4)),
        reinterpret_cast<uint32_t*>(accumulator), 0 < ki ? 1 : 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(accumulator, BlockN / 2);
}

// Issue one 2-D TMA store and nothing else.  The caller commits the bulk
// group (``T.tma_store_arrive``) and waits for it (``T.tma_store_wait``) so the
// wait can be deferred to the next tile's epilogue and overlap its mainloop.
TL_DEVICE void fp8_tma_store_2d_issue(const CUtensorMap& descriptor,
                                      void const* smem_ptr, int x, int y) {
  uint64_t desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t src = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
      "[%0, {%2, %3}], [%1];"
      :
      : "l"(desc), "r"(src), "r"(x), "r"(y)
      : "memory");
}

// Promote one 64xNx128 WGMMA partial in its per-thread accumulator layout,
// with no 2-D fragment between K blocks. The three scales arrive by value:
// the stage is already released, so nothing here may read its shared memory.
template <int BlockN>
__device__ __forceinline__ void fp8_gemm_1d2d_promote(float* partial,
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
    __nv_bfloat162_raw const v0 =
        __float22bfloat162_rn({accumulator[i * 8 + 0], accumulator[i * 8 + 1]});
    __nv_bfloat162_raw const v1 =
        __float22bfloat162_rn({accumulator[i * 8 + 2], accumulator[i * 8 + 3]});
    __nv_bfloat162_raw const v2 =
        __float22bfloat162_rn({accumulator[i * 8 + 4], accumulator[i * 8 + 5]});
    __nv_bfloat162_raw const v3 =
        __float22bfloat162_rn({accumulator[i * 8 + 6], accumulator[i * 8 + 7]});
    auto* dst = reinterpret_cast<cute::uint128_t*>(
        reinterpret_cast<__nv_bfloat16*>(output_smem) +
        (warp_in_group * 16 + lane % 16) * BlockN + i * 16 + 8 * (lane / 16));
    cute::SM90_U32x4_STSM_N::copy((uint32_t(v0.x) | (uint32_t(v0.y) << 16)),
                                  (uint32_t(v1.x) | (uint32_t(v1.y) << 16)),
                                  (uint32_t(v2.x) | (uint32_t(v2.y) << 16)),
                                  (uint32_t(v3.x) | (uint32_t(v3.y) << 16)),
                                  *dst);
  }
}

template <int BlockN, typename OutT>
__device__ __forceinline__ void fp8_gemm_raw_acc_store_global_vec2(
    float* accumulator, OutT* output, int leading_dim, int m_start, int n_start,
    int shape_m, int shape_n) {
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
        MutableOutT values[2];
        values[0] = static_cast<MutableOutT>(accumulator[i * 4 + 0]);
        values[1] = static_cast<MutableOutT>(accumulator[i * 4 + 1]);
        memcpy(&packed, values, sizeof(packed));
        *reinterpret_cast<uint32_t*>(out + row0 * leading_dim + col) = packed;
      }
      if (row1 < shape_m) {
        uint32_t packed;
        MutableOutT values[2];
        values[0] = static_cast<MutableOutT>(accumulator[i * 4 + 2]);
        values[1] = static_cast<MutableOutT>(accumulator[i * 4 + 3]);
        memcpy(&packed, values, sizeof(packed));
        *reinterpret_cast<uint32_t*>(out + row1 * leading_dim + col) = packed;
      }
    }
  }
}

#define TILEOPS_DEFINE_FP8_GEMM_1D2D_HELPERS(N)                              \
  __device__ __forceinline__ void fp8_gemm_wgmma_64x128_by_128x##N(          \
      float* acc, fp8_e4_t* a, fp8_e4_t* b) {                                \
    fp8_gemm_wgmma_64x128_by_128xN<N>(acc, a, b);                            \
  }                                                                          \
  __device__ __forceinline__ void fp8_gemm_raw_acc_stsm_bf16_64x##N(         \
      float* acc, bfloat16_t* out) {                                         \
    fp8_gemm_raw_acc_stsm_bf16<N>(acc, out);                                 \
  }                                                                          \
  template <typename OutT>                                                   \
  __device__ __forceinline__ void fp8_gemm_raw_acc_store_global_64x##N##_v2( \
      float* acc, OutT* out, int ld, int ms, int ns, int m, int n) {         \
    fp8_gemm_raw_acc_store_global_vec2<N>(acc, out, ld, ms, ns, m, n);       \
  }                                                                          \
  __device__ __forceinline__ void fp8_gemm_1d2d_promote_64x##N(              \
      float* p, float* f, float sa0, float sa1, float sb) {                  \
    fp8_gemm_1d2d_promote<N>(p, f, sa0, sa1, sb);                            \
  }

TILEOPS_DEFINE_FP8_GEMM_1D2D_HELPERS(16)
TILEOPS_DEFINE_FP8_GEMM_1D2D_HELPERS(32)
TILEOPS_DEFINE_FP8_GEMM_1D2D_HELPERS(64)
TILEOPS_DEFINE_FP8_GEMM_1D2D_HELPERS(128)

#undef TILEOPS_DEFINE_FP8_GEMM_1D2D_HELPERS

// A value every lane of the warp already holds, broadcast from lane 0 so the
// compiler can prove it warp-uniform and keep it in a uniform register.
TL_DEVICE int fp8_uniform(int value) {
  return __shfl_sync(0xffffffffu, value, 0);
}

// Prefetch one 2-D TMA box into L2 without a shared-memory destination or
// barrier.
TL_DEVICE void fp8_tma_prefetch_2d(const CUtensorMap& descriptor, int x,
                                   int y) {
  uint64_t desc = reinterpret_cast<uint64_t>(&descriptor);
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
               :
               : "l"(desc), "r"(x), "r"(y)
               : "memory");
}

// Warpgroup g's 64 rows, 64g..64g+63, of a 128-row A wave times a B stage, both
// 128B-swizzled and K-major. ``group`` is the consumer warpgroup index, which
// the caller broadcasts from lane 0 once (``fp8_uniform``), so the compiler can
// prove it warp-uniform and keeps both descriptors in uniform registers.
template <int BlockN>
__device__ __forceinline__ void fp8_wave_wgmma_64xN(float* accumulator,
                                                    fp8_e4_t* a_wave,
                                                    fp8_e4_t* b_stage,
                                                    int group) {
  fp8_gemm_wgmma_64x128_by_128xN<BlockN>(accumulator, a_wave + group * 64 * 128,
                                         b_stage);
}

// Fold one 64xN WGMMA partial into its accumulator under per-row A scales and
// per-column B scales (1x128 ``scale_b``). ``row_scale`` holds this thread's
// two rows' A scales; ``scale_b_col`` is the K-step's column of a [N][Group]
// staging of ``scale_b``, so column j's scale sits Group floats past column j
// - 1. Each column pair is read once, not once per accumulator element.
template <int BlockN, int Group>
__device__ __forceinline__ void fp8_wave_fold_per_column(
    float* final_accum, float const* partial, float const* row_scale,
    float const* scale_b_col) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    int const col = i * 8 + (lane % 4) * 2;
    float const b0 = scale_b_col[col * Group];
    float const b1 = scale_b_col[(col + 1) * Group];
    final_accum[i * 4 + 0] += partial[i * 4 + 0] * (row_scale[0] * b0);
    final_accum[i * 4 + 1] += partial[i * 4 + 1] * (row_scale[0] * b1);
    final_accum[i * 4 + 2] += partial[i * 4 + 2] * (row_scale[1] * b0);
    final_accum[i * 4 + 3] += partial[i * 4 + 3] * (row_scale[1] * b1);
  }
}

// Fold one 64xN WGMMA partial into its accumulator when the tile can span two
// 128x128 ``scale_b`` blocks: columns below ``former`` (a multiple of 8) take
// ``row_scale``, the rest ``row_scale1``. Each 8-column chunk picks its pair of
// row scales once, not once an element.
template <int BlockN>
__device__ __forceinline__ void fp8_wave_fold_split(float* final_accum,
                                                    float const* partial,
                                                    float const* row_scale,
                                                    float const* row_scale1,
                                                    int former) {
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    bool const first = i * 8 < former;
    float const s0 = first ? row_scale[0] : row_scale1[0];
    float const s1 = first ? row_scale[1] : row_scale1[1];
    final_accum[i * 4 + 0] += partial[i * 4 + 0] * s0;
    final_accum[i * 4 + 1] += partial[i * 4 + 1] * s0;
    final_accum[i * 4 + 2] += partial[i * 4 + 2] * s1;
    final_accum[i * 4 + 3] += partial[i * 4 + 3] * s1;
  }
}

__device__ __forceinline__ void fp8_wave_fold_split_56(float* f, float const* p,
                                                       float const* rs,
                                                       float const* rs1,
                                                       int former) {
  fp8_wave_fold_split<56>(f, p, rs, rs1, former);
}

// Fold one 64xN WGMMA partial into its accumulator when the tile can span three
// 128x128 ``scale_b`` blocks: columns below ``Former`` take ``row_scale``, the
// next 128 ``row_scale1``, the rest ``row_scale2``. ``Former`` is a template
// argument, so each 8-column chunk's scales are fixed at compile time; a
// runtime ``former`` costs a select or a move per chunk and K block.
template <int BlockN, int Former>
__device__ __forceinline__ void fp8_wave_fold_three(float* final_accum,
                                                    float const* partial,
                                                    float const* row_scale,
                                                    float const* row_scale1,
                                                    float const* row_scale2) {
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    float const* scale = i * 8 < Former         ? row_scale
                         : i * 8 < Former + 128 ? row_scale1
                                                : row_scale2;
    final_accum[i * 4 + 0] += partial[i * 4 + 0] * scale[0];
    final_accum[i * 4 + 1] += partial[i * 4 + 1] * scale[0];
    final_accum[i * 4 + 2] += partial[i * 4 + 2] * scale[1];
    final_accum[i * 4 + 3] += partial[i * 4 + 3] * scale[1];
  }
}

#define TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE(N, F)                   \
  __device__ __forceinline__ void fp8_wave_fold_three_##N##_##F(   \
      float* f, float const* p, float const* rs, float const* rs1, \
      float const* rs2) {                                          \
    fp8_wave_fold_three<N, F>(f, p, rs, rs1, rs2);                 \
  }

// A 176-wide tile starts at a multiple of 16 columns: its first block holds 16,
// 32, ..., or 128 of them.
TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE(176, 16)
TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE(176, 32)
TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE(176, 48)
TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE(176, 64)
TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE(176, 80)
TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE(176, 96)
TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE(176, 112)
TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE(176, 128)

#undef TILEOPS_DEFINE_FP8_WAVE_FOLD_THREE

#define TILEOPS_DEFINE_FP8_WAVE_FOLD_PER_COLUMN(N, G)                 \
  __device__ __forceinline__ void fp8_wave_fold_per_column_##N##x##G( \
      float* f, float const* p, float const* rs, float const* sb) {   \
    fp8_wave_fold_per_column<N, G>(f, p, rs, sb);                     \
  }

TILEOPS_DEFINE_FP8_WAVE_FOLD_PER_COLUMN(64, 4)
TILEOPS_DEFINE_FP8_WAVE_FOLD_PER_COLUMN(64, 8)
TILEOPS_DEFINE_FP8_WAVE_FOLD_PER_COLUMN(128, 4)
TILEOPS_DEFINE_FP8_WAVE_FOLD_PER_COLUMN(128, 8)
TILEOPS_DEFINE_FP8_WAVE_FOLD_PER_COLUMN(192, 4)
TILEOPS_DEFINE_FP8_WAVE_FOLD_PER_COLUMN(192, 8)

#undef TILEOPS_DEFINE_FP8_WAVE_FOLD_PER_COLUMN

#define TILEOPS_DEFINE_FP8_WAVE_WGMMA(N)                 \
  __device__ __forceinline__ void fp8_wave_wgmma_64x##N( \
      float* acc, fp8_e4_t* a, fp8_e4_t* b, int g) {     \
    fp8_wave_wgmma_64xN<N>(acc, a, b, g);                \
  }

TILEOPS_DEFINE_FP8_WAVE_WGMMA(56)
TILEOPS_DEFINE_FP8_WAVE_WGMMA(64)
TILEOPS_DEFINE_FP8_WAVE_WGMMA(128)
TILEOPS_DEFINE_FP8_WAVE_WGMMA(176)
TILEOPS_DEFINE_FP8_WAVE_WGMMA(192)

#undef TILEOPS_DEFINE_FP8_WAVE_WGMMA

}  // namespace tileops
