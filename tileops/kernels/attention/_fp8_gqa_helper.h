#pragma once

#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/common.h>
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

template <typename BarrierType = uint64_t>
TL_DEVICE void fp8_tma_load_2d_ptx(
    const CUtensorMap &descriptor,
    BarrierType &smem_mbar,
    void const *const smem_ptr,
    int32_t const &crd0,
    int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  uint64_t const evict_normal = 0x1000000000000000ULL;
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
      "complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(evict_normal)
      : "memory");
}

template <typename BarrierType = uint64_t>
TL_DEVICE void fp8_tma_load_4d_ptx(
    const CUtensorMap &descriptor,
    BarrierType &smem_mbar,
    void const *const smem_ptr,
    int32_t const &crd0,
    int32_t const &crd1,
    int32_t const &crd2,
    int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  uint64_t const evict_normal = 0x1000000000000000ULL;
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::"
      "complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(evict_normal)
      : "memory");
}

namespace fp8_gqa_detail {

using namespace cute;

__device__ __forceinline__ int64_t read_clock64() {
  int64_t ret;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(ret));
  return ret;
}

__device__ __forceinline__ void clock_accum_i64(int64_t* addr, int64_t delta) {
  atomicAdd(reinterpret_cast<unsigned long long*>(addr),
            static_cast<unsigned long long>(delta));
}

__device__ __forceinline__ void clock_count_i64(int64_t* addr) {
  atomicAdd(reinterpret_cast<unsigned long long*>(addr), 1ULL);
}

static constexpr int kHeadDimV = 128;
static constexpr int kBlockN = 128;
static constexpr int kBlockN224 = 224;
static constexpr int kStages = 1;

template <typename Element>
struct VTranspose128x128 {
  static constexpr cute::GMMA::Major MmaMajorV = cute::GMMA::Major::K;
  static constexpr cute::GMMA::Major TmaMajorV = cute::GMMA::Major::MN;

  // TileLang's current producer path TMA-loads V as a logical [N, D] tile into
  // shared memory.  FA3's transpose helper starts from a CUTE TMA tensor viewed
  // as [D, N].  Reinterpret the TileLang [N, D] row-major tile as that [D, N]
  // source view: physical_offset(d, n) = n * D + d.
  using SmemLayoutVt = Layout<
      Shape<Int<kHeadDimV>, Int<kBlockN>, Int<kStages>>,
      Stride<_1, Int<kHeadDimV>, _0>>;

  using SmemLayoutAtomVtMma = decltype(
      cutlass::gemm::collective::detail::ss_smem_selector<
          MmaMajorV, Element, Int<kHeadDimV>, Int<kBlockN>>());
  using SmemLayoutVtMma = decltype(tile_to_shape(
      SmemLayoutAtomVtMma{},
      make_shape(Int<kHeadDimV>{}, Int<kBlockN>{}, Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using LDSMThreadShape = Shape<_32, _4, _1, _1>;
  using LDSMThreadStride = Stride<_4, _1, _0, _0>;
  using LDSMValueShape = Shape<_2, _2, _1, _4>;
  using LDSMValueStride = Stride<_1, _2, _16, _4>;
  using LDSMDivideShape = Shape<_64, _8>;

  using S2RTiledCopyVt = decltype(make_tiled_copy(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
      Layout<LDSMThreadShape, LDSMThreadStride>{},
      Layout<LDSMValueShape, LDSMValueStride>{}));

  using STSMThreadShape = Shape<_8, _4, _4, _1>;
  using STSMThreadStride = Stride<_4, _1, _32, _0>;
  using STSMValueShape = Shape<_1, _4, _2, _2>;
  using STSMValueStride = Stride<_0, _1, _4, _8>;
  using STSMDivideShape = Shape<_8, _16>;

  using R2STiledCopyV = decltype(make_tiled_copy(
      Copy_Atom<SM90_U32x4_STSM_N, Element>{},
      Layout<STSMThreadShape, STSMThreadStride>{},
      Layout<STSMValueShape, STSMValueStride>{}));
};

template <typename Element>
struct VTranspose128x224 {
  static constexpr cute::GMMA::Major MmaMajorV = cute::GMMA::Major::K;
  static constexpr cute::GMMA::Major TmaMajorV = cute::GMMA::Major::MN;

  using SmemLayoutVt = Layout<
      Shape<Int<kHeadDimV>, Int<kBlockN224>, Int<kStages>>,
      Stride<_1, Int<kHeadDimV>, _0>>;

  using SmemLayoutAtomVtMma = decltype(
      cutlass::gemm::collective::detail::ss_smem_selector<
          MmaMajorV, Element, Int<kHeadDimV>, Int<kBlockN224>>());
  using SmemLayoutVtMma = decltype(tile_to_shape(
      SmemLayoutAtomVtMma{},
      make_shape(Int<kHeadDimV>{}, Int<kBlockN224>{}, Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using LDSMThreadShape = Shape<_32, _4, _1, _1>;
  using LDSMThreadStride = Stride<_4, _1, _0, _0>;
  using LDSMValueShape = Shape<_2, _2, _1, _4>;
  using LDSMValueStride = Stride<_1, _2, _16, _4>;
  using LDSMDivideShape = Shape<_64, _8>;

  using S2RTiledCopyVt = decltype(make_tiled_copy(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
      Layout<LDSMThreadShape, LDSMThreadStride>{},
      Layout<LDSMValueShape, LDSMValueStride>{}));

  using STSMThreadShape = Shape<_8, _4, _4, _1>;
  using STSMThreadStride = Stride<_4, _1, _32, _0>;
  using STSMValueShape = Shape<_1, _4, _2, _2>;
  using STSMValueStride = Stride<_0, _1, _4, _8>;
  using STSMDivideShape = Shape<_8, _16>;

  using R2STiledCopyV = decltype(make_tiled_copy(
      Copy_Atom<SM90_U32x4_STSM_N, Element>{},
      Layout<STSMThreadShape, STSMThreadStride>{},
      Layout<STSMValueShape, STSMValueStride>{}));
};

template <typename Element>
struct VTranspose128x128Fa3Src {
  static constexpr cute::GMMA::Major MmaMajorV = cute::GMMA::Major::K;
  static constexpr cute::GMMA::Major TmaMajorV = cute::GMMA::Major::MN;

  using SmemLayoutAtomVt = decltype(
      cutlass::gemm::collective::detail::ss_smem_selector<
          TmaMajorV, Element, Int<kHeadDimV>, Int<kBlockN>>());
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(Int<kHeadDimV>{}, Int<kBlockN>{}, Int<kStages>{}),
      Step<_2, _1, _3>{}));

  using SmemLayoutAtomVtMma = decltype(
      cutlass::gemm::collective::detail::ss_smem_selector<
          MmaMajorV, Element, Int<kHeadDimV>, Int<kBlockN>>());
  using SmemLayoutVtMma = decltype(tile_to_shape(
      SmemLayoutAtomVtMma{},
      make_shape(Int<kHeadDimV>{}, Int<kBlockN>{}, Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using LDSMThreadShape = Shape<_32, _4, _1, _1>;
  using LDSMThreadStride = Stride<_4, _1, _0, _0>;
  using LDSMValueShape = Shape<_2, _2, _1, _4>;
  using LDSMValueStride = Stride<_1, _2, _16, _4>;
  using LDSMDivideShape = Shape<_64, _8>;

  using S2RTiledCopyVt = decltype(make_tiled_copy(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
      Layout<LDSMThreadShape, LDSMThreadStride>{},
      Layout<LDSMValueShape, LDSMValueStride>{}));

  using STSMThreadShape = Shape<_8, _4, _4, _1>;
  using STSMThreadStride = Stride<_4, _1, _32, _0>;
  using STSMValueShape = Shape<_1, _4, _2, _2>;
  using STSMValueStride = Stride<_0, _1, _4, _8>;
  using STSMDivideShape = Shape<_8, _16>;

  using R2STiledCopyV = decltype(make_tiled_copy(
      Copy_Atom<SM90_U32x4_STSM_N, Element>{},
      Layout<STSMThreadShape, STSMThreadStride>{},
      Layout<STSMValueShape, STSMValueStride>{}));
};

template <typename Element>
struct VTranspose128x224Fa3Src {
  static constexpr cute::GMMA::Major MmaMajorV = cute::GMMA::Major::K;
  static constexpr cute::GMMA::Major TmaMajorV = cute::GMMA::Major::MN;

  using SmemLayoutAtomVt = decltype(
      cutlass::gemm::collective::detail::ss_smem_selector<
          TmaMajorV, Element, Int<kHeadDimV>, Int<kBlockN224>>());
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(Int<kHeadDimV>{}, Int<kBlockN224>{}, Int<kStages>{}),
      Step<_2, _1, _3>{}));

  using SmemLayoutAtomVtMma = decltype(
      cutlass::gemm::collective::detail::ss_smem_selector<
          MmaMajorV, Element, Int<kHeadDimV>, Int<kBlockN224>>());
  using SmemLayoutVtMma = decltype(tile_to_shape(
      SmemLayoutAtomVtMma{},
      make_shape(Int<kHeadDimV>{}, Int<kBlockN224>{}, Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using LDSMThreadShape = Shape<_32, _4, _1, _1>;
  using LDSMThreadStride = Stride<_4, _1, _0, _0>;
  using LDSMValueShape = Shape<_2, _2, _1, _4>;
  using LDSMValueStride = Stride<_1, _2, _16, _4>;
  using LDSMDivideShape = Shape<_64, _8>;

  using S2RTiledCopyVt = decltype(make_tiled_copy(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
      Layout<LDSMThreadShape, LDSMThreadStride>{},
      Layout<LDSMValueShape, LDSMValueStride>{}));

  using STSMThreadShape = Shape<_8, _4, _4, _1>;
  using STSMThreadStride = Stride<_4, _1, _32, _0>;
  using STSMValueShape = Shape<_1, _4, _2, _2>;
  using STSMValueStride = Stride<_0, _1, _4, _8>;
  using STSMDivideShape = Shape<_8, _16>;

  using R2STiledCopyV = decltype(make_tiled_copy(
      Copy_Atom<SM90_U32x4_STSM_N, Element>{},
      Layout<STSMThreadShape, STSMThreadStride>{},
      Layout<STSMValueShape, STSMValueStride>{}));
};

template <typename Element>
struct VTranspose128x128TlFullDst {
  using SrcConfig = VTranspose128x128<Element>;

  using SmemLayoutVt = typename SrcConfig::SmemLayoutVt;
  using SmemLayoutAtomVtMma = decltype(composition(
      Swizzle<3, 4, 3>{}, Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
  using SmemLayoutVtMma = decltype(tile_to_shape(
      SmemLayoutAtomVtMma{},
      make_shape(Int<kHeadDimV>{}, Int<kBlockN>{}, Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using LDSMThreadShape = typename SrcConfig::LDSMThreadShape;
  using LDSMThreadStride = typename SrcConfig::LDSMThreadStride;
  using LDSMValueShape = typename SrcConfig::LDSMValueShape;
  using LDSMValueStride = typename SrcConfig::LDSMValueStride;
  using LDSMDivideShape = typename SrcConfig::LDSMDivideShape;

  using STSMThreadShape = typename SrcConfig::STSMThreadShape;
  using STSMThreadStride = typename SrcConfig::STSMThreadStride;
  using STSMValueShape = typename SrcConfig::STSMValueShape;
  using STSMValueStride = typename SrcConfig::STSMValueStride;
  using STSMDivideShape = typename SrcConfig::STSMDivideShape;

  using S2RTiledCopyVt = typename SrcConfig::S2RTiledCopyVt;
  using R2STiledCopyV = typename SrcConfig::R2STiledCopyV;
};

template <typename MMA_Traits, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
  static_assert(decltype(rank<0>(acc_layout))::value == 3);
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
  if constexpr (sizeof(typename MMA_Traits::ValTypeA) == 2) {
    auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});
    return make_layout(
        make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)),
        get<1>(acc_layout),
        coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
  } else {
    static_assert(sizeof(typename MMA_Traits::ValTypeA) == 1);
    static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
    static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);
    auto l = logical_divide(get<0, 2>(acc_layout), Tile<Layout<Shape<_2, _2>>>{});
    return make_layout(
        make_layout(Layout<_4>{}, get<0, 0, 0>(l), get<0, 0, 1>(l)),
        get<1>(acc_layout),
        coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
  }
}

template <bool Transposed = false, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
  if constexpr (decltype(rank<0>(acc_layout))::value == 3) {
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = acc_layout;
    if constexpr (!Transposed) {
      return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                         make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
    } else {
      return make_layout(make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)),
                         make_layout(get<0, 1>(l), get<1>(l)));
    }
  } else {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});
    if constexpr (!Transposed) {
      return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                         make_layout(get<0, 0>(l), get<2>(l)));
    } else {
      return make_layout(make_layout(get<0, 0>(l), get<2>(l)),
                         make_layout(get<0, 1>(l), get<1>(l)));
    }
  }
}

template <typename Fragment>
CUTLASS_DEVICE void permute_Cregs_fp8(Fragment& frag) {
  Tensor frag_64b = group_modes<1, 3>(recast<uint2>(frag));
#pragma unroll
  for (int mi = 0; mi < size<1>(frag_64b); ++mi) {
#pragma unroll
    for (int i = 0; i < size<0, 2>(frag_64b) / 2; ++i) {
      auto tmp = frag_64b(make_coord(_0{}, _1{}, 2 * i), mi);
      frag_64b(make_coord(_0{}, _1{}, 2 * i), mi) =
          frag_64b(make_coord(_0{}, _0{}, 2 * i + 1), mi);
      frag_64b(make_coord(_0{}, _0{}, 2 * i + 1), mi) = tmp;
    }
  }
}

template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8(Fragment& out) {
  Tensor frag = group_modes<1, 3>(out);
#pragma unroll
  for (int mi = 0; mi < size<1>(frag); ++mi) {
#pragma unroll
    for (int j = 0; j < size<0, 1>(frag); ++j) {
#pragma unroll
      for (int i = 0; i < size<0, 2>(frag) / 2; ++i) {
        auto tmp = frag(make_coord(_1{}, j, 2 * i), mi);
        frag(make_coord(_1{}, j, 2 * i), mi) =
            frag(make_coord(_0{}, j, 2 * i + 1), mi);
        frag(make_coord(_0{}, j, 2 * i + 1), mi) = tmp;
      }
    }
  }
}

template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8_Vcolmajor(Fragment& frag) {
  int const quad_idx = static_cast<int>(threadIdx.x) & 3;
  bool const lane_03 = quad_idx == 0 || quad_idx == 3;
  static constexpr int upper_map[4] = {0, 2, 3, 1};
  using type2 = std::conditional_t<
      sizeof(typename Fragment::value_type) == 2, uint32_t, uint64_t>;
  Tensor frag_2 = group_modes<1, 3>(recast<type2>(frag));
#pragma unroll
  for (int mi = 0; mi < size<1>(frag_2); ++mi) {
#pragma unroll
    for (int j = 0; j < size<0, 1>(frag_2); ++j) {
#pragma unroll
      for (int i = 0; i < size<0, 2>(frag_2) / 2; ++i) {
        type2 upper = frag_2(make_coord(_0{}, j, 2 * i), mi);
        type2 lower = frag_2(make_coord(_0{}, j, 2 * i + 1), mi);
        type2 upper0 = lane_03 ? upper : lower;
        type2 lower0 = lane_03 ? lower : upper;
        upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
        lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 2, 4);
        frag_2(make_coord(_0{}, j, 2 * i), mi) = lane_03 ? upper0 : lower0;
        frag_2(make_coord(_0{}, j, 2 * i + 1), mi) = lane_03 ? lower0 : upper0;
      }
    }
  }
}

}  // namespace fp8_gqa_detail

template <typename FP8T>
__device__ __forceinline__ void fp8_acc_to_pv_a_frag_64x128(float* acc_s,
                                                            FP8T* p_frag,
                                                            int pv_k) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const lane_d1 = (lane >> 1) & 1;
  int const lane_b0 = lane & 1;

  #pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    int const out = pair * 2;
    int const src_rank = (lane_b0 << 1) + (pair & 1);
    int const src_base0 = pv_k * 16
                        + (((pair >> 1) & 1) * 2)
                        + (((pair >> 2) & 1) * 8);

    float const x0 = __shfl_sync(0xffffffffu, acc_s[src_base0], src_rank, 4);
    float const y0 = __shfl_sync(0xffffffffu, acc_s[src_base0 + 1], src_rank, 4);
    float const x1 = __shfl_sync(0xffffffffu, acc_s[src_base0 + 4], src_rank, 4);
    float const y1 = __shfl_sync(0xffffffffu, acc_s[src_base0 + 5], src_rank, 4);
    float const x = lane_d1 ? x1 : x0;
    float const y = lane_d1 ? y1 : y0;

    fp8_e4_2_t packed;
    float2 const xy = make_float2(x, y);
    (reinterpret_cast<__nv_fp8x2_storage_t*>(&packed))[0] =
        __nv_cvt_float2_to_fp8x2(xy, __NV_SATFINITE, __NV_E4M3);
    *(reinterpret_cast<fp8_e4_2_t*>(p_frag + out)) = packed;
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_acc_to_pv_a_frag_64x128_full_gather(
    float* acc_s, FP8T* p_frag) {
#pragma unroll
  for (int pv_k = 0; pv_k < 4; ++pv_k) {
    fp8_acc_to_pv_a_frag_64x128(acc_s, p_frag + pv_k * 16, pv_k);
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_acc_to_pv_a_frag_64x128_quad(
    float* acc_s, FP8T* p_frag, int pv_k) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const lane_d1 = (lane >> 1) & 1;
  int const lane_b0 = lane & 1;

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    int const out = pair * 2;
    int const src_rank = (lane_b0 << 1) + (pair & 1);
    int const src_base0 = pv_k * 16
                        + (((pair >> 1) & 1) * 2)
                        + (((pair >> 2) & 1) * 8);

    float const x0 = __shfl_sync(0xffffffffu, acc_s[src_base0], src_rank, 4);
    float const y0 = __shfl_sync(0xffffffffu, acc_s[src_base0 + 1], src_rank, 4);
    float const x1 = __shfl_sync(0xffffffffu, acc_s[src_base0 + 4], src_rank, 4);
    float const y1 = __shfl_sync(0xffffffffu, acc_s[src_base0 + 5], src_rank, 4);
    float const x = lane_d1 ? x1 : x0;
    float const y = lane_d1 ? y1 : y0;

    fp8_e4_2_t packed;
    float2 const xy = make_float2(x, y);
    (reinterpret_cast<__nv_fp8x2_storage_t*>(&packed))[0] =
        __nv_cvt_float2_to_fp8x2(xy, __NV_SATFINITE, __NV_E4M3);
    *(reinterpret_cast<fp8_e4_2_t*>(p_frag + out)) = packed;
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_acc_to_pv_a_frag_64x128_full_quad(
    float* acc_s, FP8T* p_frag) {
#pragma unroll
  for (int pv_k = 0; pv_k < 4; ++pv_k) {
    fp8_acc_to_pv_a_frag_64x128_quad(acc_s, p_frag + pv_k * 16, pv_k);
  }
}

__device__ __forceinline__ void fp8_permute_cregs_64x128(float* acc_s) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int const a = 8 * i + 2;
    int const b = 8 * i + 4;
    float2 tmp = *(reinterpret_cast<float2*>(acc_s + a));
    *(reinterpret_cast<float2*>(acc_s + a)) = *(reinterpret_cast<float2*>(acc_s + b));
    *(reinterpret_cast<float2*>(acc_s + b)) = tmp;
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_acc_to_pv_a_frag_64x128_fa3(float* acc_s,
                                                                FP8T* p_frag) {
#pragma unroll
  for (int pair = 0; pair < 32; ++pair) {
    fp8_e4_2_t packed;
    float2 const xy = *(reinterpret_cast<float2*>(acc_s + pair * 2));
    (reinterpret_cast<__nv_fp8x2_storage_t*>(&packed))[0] =
        __nv_cvt_float2_to_fp8x2(xy, __NV_SATFINITE, __NV_E4M3);
    *(reinterpret_cast<fp8_e4_2_t*>(p_frag + pair * 2)) = packed;
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_acc_to_pv_a_frag_64x128_cute(float* acc_s,
                                                                  FP8T* p_frag) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapePV,
                                              GMMA::Major::K, GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(
      MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(
      MmaPV{}, AtomLayout{}));

  TiledMmaQK tiled_mma_qk;
  Tensor acc_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor acc = make_tensor(acc_s, acc_template.layout());
  fp8_gqa_detail::permute_Cregs_fp8(acc);

  Tensor p_acc =
      make_tensor(acc.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(acc.layout()));
  Tensor p = make_tensor(reinterpret_cast<Element*>(p_frag), p_acc.layout());

#pragma unroll
  for (int i = 0; i < size(p_acc); ++i) {
    fp8_e4_t packed;
    (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
        __nv_cvt_float_to_fp8(p_acc(i), __NV_SATFINITE, __NV_E4M3);
    p(i) = reinterpret_cast<Element const&>(packed);
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_acc_to_pv_a_frag_64x224_cute(float* acc_s,
                                                                  FP8T* p_frag) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, Int<224>, _128>;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapePV,
                                              GMMA::Major::K, GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  TiledMmaQK tiled_mma_qk;
  Tensor acc_template = partition_fragment_C(tiled_mma_qk, Shape<_64, Int<224>>{});
  Tensor acc = make_tensor(acc_s, acc_template.layout());
  fp8_gqa_detail::permute_Cregs_fp8(acc);

  Tensor p_acc =
      make_tensor(acc.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(acc.layout()));
  Tensor p = make_tensor(reinterpret_cast<Element*>(p_frag), p_acc.layout());

#pragma unroll
  for (int i = 0; i < size(p_acc); ++i) {
    fp8_e4_t packed;
    (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
        __nv_cvt_float_to_fp8(p_acc(i), __NV_SATFINITE, __NV_E4M3);
    p(i) = reinterpret_cast<Element const&>(packed);
  }
}

__device__ __forceinline__ void fp8_qk_acc_store_global_64x224(
    float* acc_s, float q_scale, float k_scale0, float k_scale1,
    float softmax_scale, float* output, int output_row_stride, float* lse) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, Int<224>, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaQK tiled_mma_qk;
  auto thr_mma = tiled_mma_qk.get_slice(tid);
  Tensor tSrS_template = partition_fragment_C(tiled_mma_qk, Shape<_64, Int<224>>{});
  Tensor tAccS = make_tensor(acc_s, tSrS_template.layout());
  Tensor cS = make_identity_tensor(Shape<_64, Int<224>>{});
  Tensor tScS = thr_mma.partition_C(cS);

  Tensor tAccS_rowcol = make_tensor(
      tAccS.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccS.layout()));
  Tensor tScS_rowcol = make_tensor(
      tScS.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tScS.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tAccS_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tAccS_rowcol); ++n) {
    auto coord = tScS_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    float const k_scale = col < 128 ? k_scale0 : k_scale1;
    float const score = tAccS_rowcol(m, n) * q_scale * k_scale * softmax_scale;
    output[row * output_row_stride + col] = score;
    if (lse != nullptr && col == 0) {
      lse[row] = score;
    }
  }
  }
}

__device__ __forceinline__ void fp8_qk_acc_store_global_64x32_chunk(
    float* acc_s, int col_base, float q_scale, float k_scale0, float k_scale1,
    float softmax_scale, float* output, int output_row_stride, float* lse) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _32, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaQK tiled_mma_qk;
  auto thr_mma = tiled_mma_qk.get_slice(tid);
  Tensor tSrS_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _32>{});
  Tensor tAccS = make_tensor(acc_s, tSrS_template.layout());
  Tensor cS = make_identity_tensor(Shape<_64, _32>{});
  Tensor tScS = thr_mma.partition_C(cS);

  Tensor tAccS_rowcol = make_tensor(
      tAccS.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccS.layout()));
  Tensor tScS_rowcol = make_tensor(
      tScS.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tScS.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tAccS_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tAccS_rowcol); ++n) {
    auto coord = tScS_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = col_base + int(get<1>(coord));
    float const k_scale = col < 128 ? k_scale0 : k_scale1;
    float const score = tAccS_rowcol(m, n) * q_scale * k_scale * softmax_scale;
    output[row * output_row_stride + col] = score;
    if (lse != nullptr && col == 0) {
      lse[row] = score;
    }
  }
  }
}

__device__ __forceinline__ void fp8_qk_acc_store_smem_64x32_chunk(
    float* acc_s, int col_base, float q_scale, float k_scale0, float k_scale1,
    float softmax_scale, float* score_smem, int score_row_stride) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _32, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaQK tiled_mma_qk;
  auto thr_mma = tiled_mma_qk.get_slice(tid);
  Tensor tSrS_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _32>{});
  Tensor tAccS = make_tensor(acc_s, tSrS_template.layout());
  Tensor cS = make_identity_tensor(Shape<_64, _32>{});
  Tensor tScS = thr_mma.partition_C(cS);

  Tensor tAccS_rowcol = make_tensor(
      tAccS.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccS.layout()));
  Tensor tScS_rowcol = make_tensor(
      tScS.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tScS.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tAccS_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tAccS_rowcol); ++n) {
    auto coord = tScS_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = col_base + int(get<1>(coord));
    float const k_scale = col < 128 ? k_scale0 : k_scale1;
    score_smem[row * score_row_stride + col] =
        tAccS_rowcol(m, n) * q_scale * k_scale * softmax_scale;
  }
  }
}

__device__ __forceinline__ unsigned char fp8_float_to_e4m3_byte(float x) {
  fp8_e4_t packed;
  (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
      __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
  return *(reinterpret_cast<unsigned char*>(&packed));
}

__device__ __forceinline__ int fp8_fa3_p_src_index(int i) {
  int const base = i & ~7;
  int const w = i & 7;
  int const mapped = (w < 2) ? w : ((w < 4) ? (w + 2) : ((w < 6) ? (w - 2) : w));
  return base + mapped;
}

__device__ __forceinline__ uint32_t fp8_pack4_fa3_p_bytes(float* acc_s, int i) {
  float2 const lo = make_float2(acc_s[fp8_fa3_p_src_index(i + 0)],
                                acc_s[fp8_fa3_p_src_index(i + 1)]);
  float2 const hi = make_float2(acc_s[fp8_fa3_p_src_index(i + 2)],
                                acc_s[fp8_fa3_p_src_index(i + 3)]);
  __nv_fp8x2_storage_t const lo_packed =
      __nv_cvt_float2_to_fp8x2(lo, __NV_SATFINITE, __NV_E4M3);
  __nv_fp8x2_storage_t const hi_packed =
      __nv_cvt_float2_to_fp8x2(hi, __NV_SATFINITE, __NV_E4M3);
  return static_cast<uint32_t>(lo_packed) |
         (static_cast<uint32_t>(hi_packed) << 16);
}

__device__ __forceinline__ void fp8_acc_to_fa3_p_regs_64x128_no_cute(
    float* acc_s, uint32_t* p_regs) {
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    p_regs[r] = fp8_pack4_fa3_p_bytes(acc_s, r * 4);
  }
}

__device__ __forceinline__ void fp8_acc_to_fa3_p_regs_64x224_no_cute(
    float* acc_s, uint32_t* p_regs) {
#pragma unroll
  for (int r = 0; r < 28; ++r) {
    p_regs[r] = fp8_pack4_fa3_p_bytes(acc_s, r * 4);
  }
}

__device__ __forceinline__ void fp8_dump_fa3_p_pack_no_cute_vs_cute_64x128(float* out) {
  fp8_e4_t p_cute[64];
  uint32_t p_regs[16];
  float acc_s_storage[64];
  float* acc_s = acc_s_storage;

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_s[i] = (static_cast<float>(i) - 32.0f) * 0.25f;
  }

  fp8_acc_to_pv_a_frag_64x128_cute(acc_s, p_cute);

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_s[i] = (static_cast<float>(i) - 32.0f) * 0.25f;
  }
  fp8_acc_to_fa3_p_regs_64x128_no_cute(acc_s, p_regs);

  unsigned char const* p_cute_bytes = reinterpret_cast<unsigned char const*>(p_cute);
  unsigned char const* p_reg_bytes = reinterpret_cast<unsigned char const*>(p_regs);
  int const tid = static_cast<int>(threadIdx.x) & 127;
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      out[i] = static_cast<float>(p_cute_bytes[i]);
      out[64 + i] = static_cast<float>(p_reg_bytes[i]);
    }
  }
}

__device__ __forceinline__ void fp8_dump_fa3_p_pack_no_cute_vs_cute_rmem_64x128(float* out) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapePV,
                                              GMMA::Major::K, GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  float acc_s_storage[64];
  float* acc_s = acc_s_storage;
  uint32_t p_regs[16];

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_s[i] = (static_cast<float>(i) - 32.0f) * 0.25f;
  }

  TiledMmaQK tiled_mma_qk;
  Tensor acc_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor acc = make_tensor(acc_s, acc_template.layout());
  fp8_gqa_detail::permute_Cregs_fp8(acc);
  Tensor p_acc =
      make_tensor(acc.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(acc.layout()));
  Tensor tOrP = make_tensor_like<Element>(p_acc);

#pragma unroll
  for (int i = 0; i < size(tOrP); ++i) {
    fp8_e4_t packed;
    (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
        __nv_cvt_float_to_fp8(p_acc(i), __NV_SATFINITE, __NV_E4M3);
    tOrP(i) = reinterpret_cast<Element const&>(packed);
  }

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_s[i] = (static_cast<float>(i) - 32.0f) * 0.25f;
  }
  fp8_acc_to_fa3_p_regs_64x128_no_cute(acc_s, p_regs);

  Tensor tOrP_u32 = recast<uint32_t>(tOrP);
  unsigned char const* p_reg_bytes = reinterpret_cast<unsigned char const*>(p_regs);
  int const tid = static_cast<int>(threadIdx.x) & 127;
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < size(tOrP_u32); ++i) {
      uint32_t const reg = tOrP_u32(i);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        out[i * 4 + b] = static_cast<float>((reg >> (8 * b)) & 0xff);
      }
    }
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      out[64 + i] = static_cast<float>(p_reg_bytes[i]);
    }
    out[128] = static_cast<float>(size(tOrP_u32));
  }
}

__device__ __forceinline__ void fp8_dump_fa3_p_pack_no_cute_vs_cute_rmem_by_ki_64x128(float* out) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapePV,
                                              GMMA::Major::K, GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  float acc_s_storage[64];
  float* acc_s = acc_s_storage;
  uint32_t p_regs[16];

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_s[i] = (static_cast<float>(i) - 32.0f) * 0.25f;
  }

  TiledMmaQK tiled_mma_qk;
  Tensor acc_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor acc = make_tensor(acc_s, acc_template.layout());
  fp8_gqa_detail::permute_Cregs_fp8(acc);
  Tensor p_acc =
      make_tensor(acc.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(acc.layout()));
  Tensor tOrP = make_tensor_like<Element>(p_acc);

#pragma unroll
  for (int i = 0; i < size(tOrP); ++i) {
    fp8_e4_t packed;
    (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
        __nv_cvt_float_to_fp8(p_acc(i), __NV_SATFINITE, __NV_E4M3);
    tOrP(i) = reinterpret_cast<Element const&>(packed);
  }

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_s[i] = (static_cast<float>(i) - 32.0f) * 0.25f;
  }
  fp8_acc_to_fa3_p_regs_64x128_no_cute(acc_s, p_regs);

  int const tid = static_cast<int>(threadIdx.x) & 127;
  if (tid == 0) {
#pragma unroll
    for (int ki = 0; ki < size<2>(tOrP); ++ki) {
      Tensor tOrPKiU32 = recast<uint32_t>(tOrP(_, _, ki));
#pragma unroll
      for (int r = 0; r < size(tOrPKiU32); ++r) {
        uint32_t const reg = tOrPKiU32(r);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          out[ki * 32 + r * 4 + b] = static_cast<float>((reg >> (8 * b)) & 0xff);
        }
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        uint32_t const reg = p_regs[ki * 4 + r];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          out[ki * 32 + 16 + r * 4 + b] = static_cast<float>((reg >> (8 * b)) & 0xff);
        }
      }
    }
    out[128] = static_cast<float>(size<2>(tOrP));
  }
}

__device__ __forceinline__ void fp8_dump_fa3_p_areg_src_64x128(float* out) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapePV,
                                              GMMA::Major::K, GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  float acc_s_storage[64];
  float* acc_s = acc_s_storage;

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_s[i] = static_cast<float>(i);
  }

  TiledMmaQK tiled_mma_qk;
  Tensor acc_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor acc = make_tensor(acc_s, acc_template.layout());
  fp8_gqa_detail::permute_Cregs_fp8(acc);

  Tensor p_acc =
      make_tensor(acc.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(acc.layout()));

#pragma unroll
  for (int i = 0; i < size(p_acc); ++i) {
    out[tid * 65 + i] = p_acc(i);
  }
  out[tid * 65 + 64] = static_cast<float>(size(p_acc));
}

__device__ __forceinline__ void fp8_dump_fa3_p_areg_src_by_ki_64x128(float* out) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapePV,
                                              GMMA::Major::K, GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  float acc_s_storage[64];
  float* acc_s = acc_s_storage;

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_s[i] = static_cast<float>(i);
  }

  TiledMmaQK tiled_mma_qk;
  Tensor acc_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor acc = make_tensor(acc_s, acc_template.layout());
  fp8_gqa_detail::permute_Cregs_fp8(acc);

  Tensor p_acc =
      make_tensor(acc.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(acc.layout()));

#pragma unroll
  for (int ki = 0; ki < size<2>(p_acc); ++ki) {
    Tensor p_acc_ki = p_acc(_, _, ki);
#pragma unroll
    for (int i = 0; i < size(p_acc_ki); ++i) {
      out[tid * 65 + ki * 16 + i] = p_acc_ki(i);
    }
  }
  out[tid * 65 + 64] = static_cast<float>(size<2>(p_acc));
}

__device__ __forceinline__ void fp8_permute_output_64x128(float* acc_o) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int const a = 8 * i + 2 * j + 1;
      int const b = 8 * i + 2 * j + 4;
      float tmp = acc_o[a];
      acc_o[a] = acc_o[b];
      acc_o[b] = tmp;
    }
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x128_ldsm_stsm(FP8T* v_smem,
                                                                  FP8T* v_tc_smem) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x128<FP8T>;

  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_smem), typename Config::SmemLayoutVt{}));
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));

  int const thread_idx = static_cast<int>(threadIdx.x) & 127;
  typename Config::S2RTiledCopyVt s2r_tiled_copy_vt;
  typename Config::R2STiledCopyV r2s_tiled_copy_v;
  auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
  auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);

  Tensor tTranssVt_ =
      s2r_thr_copy_vt.partition_S(flat_divide(sVt, typename Config::LDSMDivideShape{}));
  Tensor tTranssV_ =
      r2s_thr_copy_v.partition_D(flat_divide(sV, typename Config::STSMDivideShape{}));

  static constexpr int TransposeILP =
      (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
  Tensor tTranssVt = logical_divide(
      group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
      Shape<Underscore, Int<TransposeILP>>{});
  Tensor tTranssV = logical_divide(
      group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
      Shape<Underscore, Int<TransposeILP>>{});

#pragma unroll
  for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
    Tensor tTransrV64 = recast<uint2>(tTransrV);
    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), _0{}), tTransrV);
#pragma unroll
    for (int j = 0; j < size(tTransrV64); ++j) {
      uint32_t upper = tTransrV64[j].x;
      uint32_t lower = tTransrV64[j].y;
      tTransrV64[j].x = __byte_perm(upper, lower, 0x6420);
      tTransrV64[j].y = __byte_perm(upper, lower, 0x7531);
    }
    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), _0{}));
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x224_ldsm_stsm(FP8T* v_smem,
                                                                  FP8T* v_tc_smem) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x224<FP8T>;

  Tensor sVt = make_tensor(make_smem_ptr(v_smem), typename Config::SmemLayoutVt{});
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));

  int const thread_idx = static_cast<int>(threadIdx.x) & 127;
  typename Config::S2RTiledCopyVt s2r_tiled_copy_vt;
  typename Config::R2STiledCopyV r2s_tiled_copy_v;
  auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
  auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);

  Tensor tTranssVt_ =
      s2r_thr_copy_vt.partition_S(flat_divide(sVt, typename Config::LDSMDivideShape{}));
  Tensor tTranssV_ =
      r2s_thr_copy_v.partition_D(flat_divide(sV, typename Config::STSMDivideShape{}));

  static constexpr int TransposeILP =
      (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
  Tensor tTranssVt = logical_divide(
      group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
      Shape<Underscore, Int<TransposeILP>>{});
  Tensor tTranssV = logical_divide(
      group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
      Shape<Underscore, Int<TransposeILP>>{});

#pragma unroll
  for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
    Tensor tTransrV64 = recast<uint2>(tTransrV);
    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), _0{}), tTransrV);
#pragma unroll
    for (int j = 0; j < size(tTransrV64); ++j) {
      uint32_t upper = tTransrV64[j].x;
      uint32_t lower = tTransrV64[j].y;
      tTransrV64[j].x = __byte_perm(upper, lower, 0x6420);
      tTransrV64[j].y = __byte_perm(upper, lower, 0x7531);
    }
    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), _0{}));
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x128_ldsm_stsm_no_prmt(
    FP8T* v_smem, FP8T* v_tc_smem) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x128<FP8T>;

  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_smem), typename Config::SmemLayoutVt{}));
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));

  int const thread_idx = static_cast<int>(threadIdx.x) & 127;
  typename Config::S2RTiledCopyVt s2r_tiled_copy_vt;
  typename Config::R2STiledCopyV r2s_tiled_copy_v;
  auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
  auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);

  Tensor tTranssVt_ =
      s2r_thr_copy_vt.partition_S(flat_divide(sVt, typename Config::LDSMDivideShape{}));
  Tensor tTranssV_ =
      r2s_thr_copy_v.partition_D(flat_divide(sV, typename Config::STSMDivideShape{}));

  static constexpr int TransposeILP =
      (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
  Tensor tTranssVt = logical_divide(
      group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
      Shape<Underscore, Int<TransposeILP>>{});
  Tensor tTranssV = logical_divide(
      group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
      Shape<Underscore, Int<TransposeILP>>{});

#pragma unroll
  for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), _0{}), tTransrV);
    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), _0{}));
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x128_ldsm_stsm_tl_full(
    FP8T* v_smem, FP8T* v_tc_smem) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x128TlFullDst<FP8T>;

  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_smem), typename Config::SmemLayoutVt{}));
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));

  int const thread_idx = static_cast<int>(threadIdx.x) & 127;
  typename Config::S2RTiledCopyVt s2r_tiled_copy_vt;
  typename Config::R2STiledCopyV r2s_tiled_copy_v;
  auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
  auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);

  Tensor tTranssVt_ =
      s2r_thr_copy_vt.partition_S(flat_divide(sVt, typename Config::LDSMDivideShape{}));
  Tensor tTranssV_ =
      r2s_thr_copy_v.partition_D(flat_divide(sV, typename Config::STSMDivideShape{}));

  static constexpr int TransposeILP =
      (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
  Tensor tTranssVt = logical_divide(
      group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
      Shape<Underscore, Int<TransposeILP>>{});
  Tensor tTranssV = logical_divide(
      group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
      Shape<Underscore, Int<TransposeILP>>{});

#pragma unroll
  for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
    Tensor tTransrV64 = recast<uint2>(tTransrV);
    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), _0{}), tTransrV);
#pragma unroll
    for (int j = 0; j < size(tTransrV64); ++j) {
      uint32_t upper = tTransrV64[j].x;
      uint32_t lower = tTransrV64[j].y;
      tTransrV64[j].x = __byte_perm(upper, lower, 0x6420);
      tTransrV64[j].y = __byte_perm(upper, lower, 0x7531);
    }
    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), _0{}));
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pack_v_128x128_tl_ldsm_src(
    FP8T* v_smem, FP8T* v_perm_smem) {
  int const tid = static_cast<int>(threadIdx.x) & 127;

#pragma unroll
  for (int d = 0; d < 128; ++d) {
    int const n = tid;
    int const n_within = n & 15;
    int const d_within = d & 15;
    int const perm_n = (n & ~15)
                     + (n_within & 1)
                     + ((n_within & 2) << 2)
                     + ((n_within & 4) >> 1)
                     + ((n_within & 8) >> 1);
    int const perm_d = (d & ~15) + ((d_within & 7) << 1) + (d_within >> 3);
    v_perm_smem[perm_n * 128 + perm_d] = v_smem[n * 128 + d];
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_dump_128x128_raw(FP8T* smem, FP8T* out) {
  int const tid = static_cast<int>(threadIdx.x) & 127;

#pragma unroll
  for (int i = 0; i < 128; ++i) {
    int const offset = tid * 128 + i;
    out[offset] = smem[offset];
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_dump_128x224_raw(FP8T* smem, FP8T* out) {
  int const tid = static_cast<int>(threadIdx.x) & 127;

#pragma unroll
  for (int i = 0; i < 224; ++i) {
    int const offset = tid * 224 + i;
    out[offset] = smem[offset];
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_keep_smem(FP8T* smem) {
  asm volatile("" : : "l"(smem) : "memory");
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x128_pack_ldsm_stsm_tl_full(
    FP8T* v_smem, FP8T* v_perm_smem, FP8T* v_tc_smem) {
  fp8_pack_v_128x128_tl_ldsm_src(v_smem, v_perm_smem);
  __syncthreads();
  fp8_transpose_v_128x128_ldsm_stsm(v_perm_smem, v_tc_smem);
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pack_v_128x128_fa3_vt(FP8T* v_smem,
                                                          FP8T* v_vt_smem) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_vt_smem), typename Config::SmemLayoutVt{}));
  int const tid = static_cast<int>(threadIdx.x) & 127;

#pragma unroll
  for (int d = 0; d < 128; ++d) {
    int const n = tid;
    sVt(d, n, 0) = v_smem[n * 128 + d];
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x128_fa3_src_ldsm_stsm(
    FP8T* v_vt_smem, FP8T* v_tc_smem) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_vt_smem), typename Config::SmemLayoutVt{}));
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));

  int const thread_idx = static_cast<int>(threadIdx.x) & 127;
  typename Config::S2RTiledCopyVt s2r_tiled_copy_vt;
  typename Config::R2STiledCopyV r2s_tiled_copy_v;
  auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
  auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);

  Tensor tTranssVt_ =
      s2r_thr_copy_vt.partition_S(flat_divide(sVt, typename Config::LDSMDivideShape{}));
  Tensor tTranssV_ =
      r2s_thr_copy_v.partition_D(flat_divide(sV, typename Config::STSMDivideShape{}));

  static constexpr int TransposeILP =
      (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
  Tensor tTranssVt = logical_divide(
      group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
      Shape<Underscore, Int<TransposeILP>>{});
  Tensor tTranssV = logical_divide(
      group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
      Shape<Underscore, Int<TransposeILP>>{});

#pragma unroll
  for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
    Tensor tTransrV64 = recast<uint2>(tTransrV);
    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), _0{}), tTransrV);
#pragma unroll
    for (int j = 0; j < size(tTransrV64); ++j) {
      uint32_t upper = tTransrV64[j].x;
      uint32_t lower = tTransrV64[j].y;
      tTransrV64[j].x = __byte_perm(upper, lower, 0x6420);
      tTransrV64[j].y = __byte_perm(upper, lower, 0x7531);
    }
    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), _0{}));
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x224_fa3_src_ldsm_stsm(
    FP8T* v_vt_smem, FP8T* v_tc_smem) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x224Fa3Src<FP8T>;

  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_vt_smem), typename Config::SmemLayoutVt{}));
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));

  int const thread_idx = static_cast<int>(threadIdx.x) & 127;
  typename Config::S2RTiledCopyVt s2r_tiled_copy_vt;
  typename Config::R2STiledCopyV r2s_tiled_copy_v;
  auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
  auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);

  Tensor tTranssVt_ =
      s2r_thr_copy_vt.partition_S(flat_divide(sVt, typename Config::LDSMDivideShape{}));
  Tensor tTranssV_ =
      r2s_thr_copy_v.partition_D(flat_divide(sV, typename Config::STSMDivideShape{}));

  static constexpr int TransposeILP =
      (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
  Tensor tTranssVt = logical_divide(
      group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
      Shape<Underscore, Int<TransposeILP>>{});
  Tensor tTranssV = logical_divide(
      group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
      Shape<Underscore, Int<TransposeILP>>{});

#pragma unroll
  for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
    Tensor tTransrV64 = recast<uint2>(tTransrV);
    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), _0{}), tTransrV);
#pragma unroll
    for (int j = 0; j < size(tTransrV64); ++j) {
      uint32_t upper = tTransrV64[j].x;
      uint32_t lower = tTransrV64[j].y;
      tTransrV64[j].x = __byte_perm(upper, lower, 0x6420);
      tTransrV64[j].y = __byte_perm(upper, lower, 0x7531);
    }
    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), _0{}));
  }
}

__device__ __forceinline__ void fp8_producer_barrier_128() {
  asm volatile("bar.sync 15, 128;\n" ::: "memory");
}

template <typename FP8T>
__device__ __noinline__ void fp8_transpose_v_128x224_fa3_src_ldsm_stsm_noinline_fence(
    FP8T* v_vt_smem, FP8T* v_tc_smem) {
  asm volatile("" ::: "memory");
  fp8_transpose_v_128x224_fa3_src_ldsm_stsm(v_vt_smem, v_tc_smem);
  asm volatile("" ::: "memory");
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x224_fa3_src_ldsm_stsm_barrier_each_iter(
    FP8T* v_vt_smem, FP8T* v_tc_smem) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x224Fa3Src<FP8T>;

  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_vt_smem), typename Config::SmemLayoutVt{}));
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));

  int const thread_idx = static_cast<int>(threadIdx.x) & 127;
  typename Config::S2RTiledCopyVt s2r_tiled_copy_vt;
  typename Config::R2STiledCopyV r2s_tiled_copy_v;
  auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
  auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);

  Tensor tTranssVt_ =
      s2r_thr_copy_vt.partition_S(flat_divide(sVt, typename Config::LDSMDivideShape{}));
  Tensor tTranssV_ =
      r2s_thr_copy_v.partition_D(flat_divide(sV, typename Config::STSMDivideShape{}));

  static constexpr int TransposeILP =
      (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
  Tensor tTranssVt = logical_divide(
      group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
      Shape<Underscore, Int<TransposeILP>>{});
  Tensor tTranssV = logical_divide(
      group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
      Shape<Underscore, Int<TransposeILP>>{});

#pragma unroll
  for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
    Tensor tTransrV64 = recast<uint2>(tTransrV);
    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), _0{}), tTransrV);
    fp8_producer_barrier_128();
#pragma unroll
    for (int j = 0; j < size(tTransrV64); ++j) {
      uint32_t upper = tTransrV64[j].x;
      uint32_t lower = tTransrV64[j].y;
      tTransrV64[j].x = __byte_perm(upper, lower, 0x6420);
      tTransrV64[j].y = __byte_perm(upper, lower, 0x7531);
    }
    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), _0{}));
    fp8_producer_barrier_128();
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x224_fa3_src_ldsm_stsm_load_all_store_all(
    FP8T* v_vt_smem, FP8T* v_tc_smem) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x224Fa3Src<FP8T>;

  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_vt_smem), typename Config::SmemLayoutVt{}));
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));

  int const thread_idx = static_cast<int>(threadIdx.x) & 127;
  typename Config::S2RTiledCopyVt s2r_tiled_copy_vt;
  typename Config::R2STiledCopyV r2s_tiled_copy_v;
  auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
  auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);

  Tensor tTranssVt_ =
      s2r_thr_copy_vt.partition_S(flat_divide(sVt, typename Config::LDSMDivideShape{}));
  Tensor tTranssV_ =
      r2s_thr_copy_v.partition_D(flat_divide(sV, typename Config::STSMDivideShape{}));

  static constexpr int TransposeILP =
      (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
  Tensor tTranssVt = logical_divide(
      group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
      Shape<Underscore, Int<TransposeILP>>{});
  Tensor tTranssV = logical_divide(
      group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
      Shape<Underscore, Int<TransposeILP>>{});

  Tensor tTransrV = make_fragment_like(tTranssV);
  Tensor tTransrV64 = recast<uint2>(tTransrV);
  cute::copy(s2r_tiled_copy_vt, tTranssVt, tTransrV);
  fp8_producer_barrier_128();

#pragma unroll
  for (int j = 0; j < size(tTransrV64); ++j) {
    uint32_t upper = tTransrV64[j].x;
    uint32_t lower = tTransrV64[j].y;
    tTransrV64[j].x = __byte_perm(upper, lower, 0x6420);
    tTransrV64[j].y = __byte_perm(upper, lower, 0x7531);
  }

  cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV);
  fp8_producer_barrier_128();
}

template <typename FP8T>
__device__ __forceinline__ void fp8_store_v_tc_direct_128x224(
    FP8T* v_tc_smem, int d, int n, FP8T value) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x224<FP8T>;
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));
  sV(d, n, 0) = value;
}

template <typename FP8T>
__device__ __forceinline__ void fp8_transpose_v_128x128_tl_full_swizzle(
    FP8T* v_smem, FP8T* v_tc_smem) {
  int const tid = static_cast<int>(threadIdx.x) & 127;

#pragma unroll
  for (int d = 0; d < 128; ++d) {
    int const n = tid;
    int const src = n * 128 + d;
    int const dst = d * 128 + (((n >> 4) ^ (d & 7)) << 4) + (n & 15);
    v_tc_smem[dst] = v_smem[src];
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_cute_unit_64x128x128(
    FP8T* p_smem, FP8T* v_smem, FP8T* v_vt_smem, FP8T* v_tc_smem,
    int direct_v, float* o_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  if ((direct_v & 1) != 0) {
    Tensor sV_direct = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{}));
    int const tid = static_cast<int>(threadIdx.x) & 127;
#pragma unroll
    for (int d = 0; d < 128; ++d) {
      int const n = tid;
      sV_direct(d, n, 0) = reinterpret_cast<Element const&>(v_smem[n * 128 + d]);
    }
  } else {
    fp8_pack_v_128x128_fa3_vt(v_smem, v_vt_smem);
    __syncthreads();
    fp8_transpose_v_128x128_fa3_src_ldsm_stsm(v_vt_smem, v_tc_smem);
  }
  __syncthreads();

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_slice(static_cast<int>(threadIdx.x) & 127);
  auto thr_mma = tiled_mma_pv.get_slice(static_cast<int>(threadIdx.x) & 127);

  Tensor sP = make_tensor(make_smem_ptr(p_smem), Layout<Shape<_64, _128>, Stride<_128, _1>>{});
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor cP = make_identity_tensor(Shape<_64, _128>{});
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});

  Tensor tOsP = thr_mma.partition_A(sP);
  Tensor tOrP = thr_mma.make_fragment_A(tOsP);
  Tensor tSrS = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor tScP = thr_mma_qk.partition_C(cP);
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});
  Tensor tOrO = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((direct_v & 4) != 0) {
    Tensor tSrS_rowcol = make_tensor(
        tSrS.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tSrS.layout()));
    Tensor tScP_rowcol = make_tensor(
        tScP.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tScP.layout()));
#pragma unroll
    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
#pragma unroll
    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
      auto coord = tScP_rowcol(m, n);
      int const row = int(get<0>(coord));
      int const col = int(get<1>(coord));
      tSrS_rowcol(m, n) = static_cast<float>(
          reinterpret_cast<Element const&>(p_smem[row * 128 + col]));
    }
    }
    fp8_gqa_detail::permute_Cregs_fp8(tSrS);
    Tensor tOrP_acc = make_tensor(
        tSrS.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
#pragma unroll
    for (int i = 0; i < size(tOrP); ++i) {
      fp8_e4_t packed;
      (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
          __nv_cvt_float_to_fp8(tOrP_acc(i), __NV_SATFINITE, __NV_E4M3);
      tOrP(i) = reinterpret_cast<Element const&>(packed);
    }
  } else {
    cute::copy(tOsP, tOrP);
  }

  clear(tOrO);
  warpgroup_fence_operand(tOrP);
  warpgroup_fence_operand(tOrO);
  warpgroup_arrive();
  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

#pragma unroll
  for (int ki = 0; ki < size<2>(tOrP); ++ki) {
    cute::gemm(tiled_mma_pv, tOrP(_, _, ki), tOrV(_, _, ki), tOrO);
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
  }

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tOrO);
  warpgroup_fence_operand(tOrP);

  if ((direct_v & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tOrO);
  }
  Tensor tOrO_rowcol = make_tensor(
      tOrO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOrO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tOrO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tOrO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    int store_col = col;
    if ((direct_v & 4) != 0 && (direct_v & 2) == 0) {
      int const base = col & ~15;
      int const within = col & 15;
      int const pair = within >> 1;
      int const low = within & 1;
      int const mapped_pair = ((pair & 3) << 1) | (pair >> 2);
      store_col = base + (mapped_pair << 1) + low;
    }
    o_smem[row * 128 + store_col] = tOrO_rowcol(m, n);
  }
  }
  __syncthreads();
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_cute_unit_from_acc_64x128x128(
    float* acc_s, FP8T* v_smem, FP8T* v_vt_smem, FP8T* v_tc_smem,
    int flags, float* o_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  if ((flags & 1) != 0) {
    Tensor sV_direct = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{}));
    int const tid = static_cast<int>(threadIdx.x) & 127;
#pragma unroll
    for (int d = 0; d < 128; ++d) {
      int const n = tid;
      sV_direct(d, n, 0) = reinterpret_cast<Element const&>(v_smem[n * 128 + d]);
    }
  } else {
    fp8_pack_v_128x128_fa3_vt(v_smem, v_vt_smem);
    __syncthreads();
    fp8_transpose_v_128x128_fa3_src_ldsm_stsm(v_vt_smem, v_tc_smem);
  }
  __syncthreads();

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(static_cast<int>(threadIdx.x) & 127);

  Tensor tSrS_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor tSrS = make_tensor(acc_s, tSrS_template.layout());
  Tensor tOrP_acc = make_tensor(
      tSrS.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
  Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});
  Tensor tOrO = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 4) != 0) {
    fp8_gqa_detail::permute_Cregs_fp8(tSrS);
  }
#pragma unroll
  for (int i = 0; i < size(tOrP); ++i) {
    fp8_e4_t packed;
    (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
        __nv_cvt_float_to_fp8(tOrP_acc(i), __NV_SATFINITE, __NV_E4M3);
    tOrP(i) = reinterpret_cast<Element const&>(packed);
  }

  clear(tOrO);
  warpgroup_fence_operand(tOrP);
  warpgroup_fence_operand(tOrO);
  warpgroup_arrive();
  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

#pragma unroll
  for (int ki = 0; ki < size<2>(tOrP); ++ki) {
    cute::gemm(tiled_mma_pv, tOrP(_, _, ki), tOrV(_, _, ki), tOrO);
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
  }

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tOrO);
  warpgroup_fence_operand(tOrP);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tOrO);
  }
  Tensor tOrO_rowcol = make_tensor(
      tOrO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOrO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tOrO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tOrO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    int store_col = col;
    if ((flags & 4) != 0 && (flags & 2) == 0) {
      int const base = col & ~15;
      int const within = col & 15;
      int const pair = within >> 1;
      int const low = within & 1;
      int const mapped_pair = ((pair & 3) << 1) | (pair >> 2);
      store_col = base + (mapped_pair << 1) + low;
    }
    o_smem[row * 128 + store_col] = tOrO_rowcol(m, n);
  }
  }
  __syncthreads();
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_cute_unit_from_acc_pretransposed_64x128x128(
    float* acc_s, FP8T* v_tc_smem, int flags, float* o_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(static_cast<int>(threadIdx.x) & 127);

  Tensor tSrS_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor tSrS = make_tensor(acc_s, tSrS_template.layout());
  Tensor tOrP_acc = make_tensor(
      tSrS.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
  Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});
  Tensor tOrO = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 4) != 0) {
    fp8_gqa_detail::permute_Cregs_fp8(tSrS);
  }
#pragma unroll
  for (int i = 0; i < size(tOrP); ++i) {
    fp8_e4_t packed;
    (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
        __nv_cvt_float_to_fp8(tOrP_acc(i), __NV_SATFINITE, __NV_E4M3);
    tOrP(i) = reinterpret_cast<Element const&>(packed);
  }

  clear(tOrO);
  warpgroup_fence_operand(tOrP);
  warpgroup_fence_operand(tOrO);
  warpgroup_arrive();
  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

#pragma unroll
  for (int ki = 0; ki < size<2>(tOrP); ++ki) {
    cute::gemm(tiled_mma_pv, tOrP(_, _, ki), tOrV(_, _, ki), tOrO);
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
  }

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tOrO);
  warpgroup_fence_operand(tOrP);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tOrO);
  }
  Tensor tOrO_rowcol = make_tensor(
      tOrO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOrO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tOrO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tOrO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    int store_col = col;
    if ((flags & 4) != 0 && (flags & 2) == 0) {
      int const base = col & ~15;
      int const within = col & 15;
      int const pair = within >> 1;
      int const low = within & 1;
      int const mapped_pair = ((pair & 3) << 1) | (pair >> 2);
      store_col = base + (mapped_pair << 1) + low;
    }
    o_smem[row * 128 + store_col] = tOrO_rowcol(m, n);
  }
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_cute_unit_from_acc_pretransposed_to_acc_64x128x128(
    float* acc_s, FP8T* v_tc_smem, int flags, float* acc_delta) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(static_cast<int>(threadIdx.x) & 127);

  Tensor tSrS_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor tSrS = make_tensor(acc_s, tSrS_template.layout());
  Tensor tOrP_acc = make_tensor(
      tSrS.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
  Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});
  Tensor tOrO = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});

  if ((flags & 4) != 0) {
    fp8_gqa_detail::permute_Cregs_fp8(tSrS);
  }
#pragma unroll
  for (int i = 0; i < size(tOrP); ++i) {
    fp8_e4_t packed;
    (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
        __nv_cvt_float_to_fp8(tOrP_acc(i), __NV_SATFINITE, __NV_E4M3);
    tOrP(i) = reinterpret_cast<Element const&>(packed);
  }

  clear(tOrO);
  warpgroup_fence_operand(tOrP);
  warpgroup_fence_operand(tOrO);
  warpgroup_arrive();
  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

#pragma unroll
  for (int ki = 0; ki < size<2>(tOrP); ++ki) {
    cute::gemm(tiled_mma_pv, tOrP(_, _, ki), tOrV(_, _, ki), tOrO);
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
  }

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tOrO);
  warpgroup_fence_operand(tOrP);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tOrO);
  }

  Tensor tOut = make_tensor(acc_delta, tOrO.layout());
#pragma unroll
  for (int i = 0; i < size(tOrO); ++i) {
    tOut(i) = tOrO(i);
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_ptx_unit_from_acc_pretransposed_to_acc_64x128x128(
    float* acc_s, FP8T* v_tc_smem, int flags, float* acc_delta) {
  uint32_t p_regs[16];
  GmmaDescriptor desc_b;
  initialize_wgmma_descriptor<1, 1, 64>(desc_b, v_tc_smem);

  fp8_acc_to_fa3_p_regs_64x128_no_cute(acc_s, p_regs);

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b + ((ki * 32) >> 4)),
        reinterpret_cast<uint32_t*>(acc_delta),
        ki != 0);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  if ((flags & 2) == 0) {
    fp8_permute_output_64x128(acc_delta);
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_ptx_unit_from_acc_pretransposed_to_smem_64x128x128(
    float* acc_s, FP8T* v_tc_smem, int flags, float* o_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  uint32_t p_regs[16];
  float o_ptx_storage[64];
  float* o_ptx = o_ptx_storage;
  tl::GmmaDescriptor desc_b;

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    o_ptx[i] = 0.0f;
  }

  initialize_wgmma_descriptor<1, 1, 64>(desc_b, v_tc_smem);
  fp8_acc_to_fa3_p_regs_64x128_no_cute(acc_s, p_regs);

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b + ((ki * 32) >> 4)),
        reinterpret_cast<uint32_t*>(o_ptx),
        ki != 0);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tOrO = make_tensor(o_ptx, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tOrO);
  }

  Tensor tOrO_rowcol = make_tensor(
      tOrO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOrO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tOrO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tOrO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    int store_col = col;
    if ((flags & 4) != 0 && (flags & 2) == 0) {
      int const base = col & ~15;
      int const within = col & 15;
      int const pair = within >> 1;
      int const low = within & 1;
      int const mapped_pair = ((pair & 3) << 1) | (pair >> 2);
      store_col = base + (mapped_pair << 1) + low;
    }
    o_smem[row * 128 + store_col] = tOrO_rowcol(m, n);
  }
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x128(
    float* acc_s, FP8T* v_tc_smem, int flags, float* ss, float v_scale, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  uint32_t p_regs[16];
  float delta_storage[64];
  float* delta = delta_storage;
  tl::GmmaDescriptor desc_b;

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    delta[i] = 0.0f;
  }

  initialize_wgmma_descriptor<1, 1, 64>(desc_b, v_tc_smem);
  fp8_acc_to_fa3_p_regs_64x128_no_cute(acc_s, p_regs);

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b + ((ki * 32) >> 4)),
        reinterpret_cast<uint32_t*>(delta),
        ki != 0);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tDelta = make_tensor(delta, tOrO_template.layout());
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tDelta);
  }

  Tensor tDelta_rowcol = make_tensor(
      tDelta.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tDelta.layout()));
  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tDelta_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tDelta_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    tAccO_rowcol(m, n) = tAccO_rowcol(m, n) * ss[row] + tDelta_rowcol(m, n) * v_scale;
  }
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224(
    float* acc_s, FP8T* v_tc_smem, int flags, float* ss, float v_scale, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x224<FP8T>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  uint32_t p_regs[28];
  float delta_storage[64];
  float* delta = delta_storage;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    delta[i] = 0.0f;
  }

  fp8_acc_to_fa3_p_regs_64x224_no_cute(acc_s, p_regs);

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 7; ++ki) {
    cute::GmmaDescriptor desc_b = tOrV(_, _, ki)(0);
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b),
        reinterpret_cast<uint32_t*>(delta),
        ki != 0);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tDelta = make_tensor(delta, tOrO_template.layout());
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tDelta);
  }

  Tensor tDelta_rowcol = make_tensor(
      tDelta.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tDelta.layout()));
  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tDelta_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tDelta_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    tAccO_rowcol(m, n) = tAccO_rowcol(m, n) * ss[row] + tDelta_rowcol(m, n) * v_scale;
  }
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224_timed(
    float* acc_s, FP8T* v_tc_smem, int flags, float* ss, float v_scale, float* acc_o,
    int64_t* timing) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x224<FP8T>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  uint32_t p_regs[28];
  float delta_storage[64];
  float* delta = delta_storage;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});

  int64_t const t0 = fp8_gqa_detail::read_clock64();
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    delta[i] = 0.0f;
  }
  int64_t const t1 = fp8_gqa_detail::read_clock64();

  fp8_acc_to_fa3_p_regs_64x224_no_cute(acc_s, p_regs);
  int64_t const t2 = fp8_gqa_detail::read_clock64();

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 7; ++ki) {
    cute::GmmaDescriptor desc_b = tOrV(_, _, ki)(0);
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b),
        reinterpret_cast<uint32_t*>(delta),
        ki != 0);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  int64_t const t3 = fp8_gqa_detail::read_clock64();

  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  int64_t const t4 = fp8_gqa_detail::read_clock64();

  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tDelta = make_tensor(delta, tOrO_template.layout());
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tDelta);
  }

  Tensor tDelta_rowcol = make_tensor(
      tDelta.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tDelta.layout()));
  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tDelta_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tDelta_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    tAccO_rowcol(m, n) = tAccO_rowcol(m, n) * ss[row] + tDelta_rowcol(m, n) * v_scale;
  }
  }
  int64_t const t5 = fp8_gqa_detail::read_clock64();

  if (tid == 0) {
    fp8_gqa_detail::clock_accum_i64(timing + 20, t1 - t0);
    fp8_gqa_detail::clock_accum_i64(timing + 21, t2 - t1);
    fp8_gqa_detail::clock_accum_i64(timing + 22, t3 - t2);
    fp8_gqa_detail::clock_accum_i64(timing + 23, t4 - t3);
    fp8_gqa_detail::clock_accum_i64(timing + 24, t5 - t4);
    fp8_gqa_detail::clock_accum_i64(timing + 25, t5 - t0);
    fp8_gqa_detail::clock_count_i64(timing + 26);
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pack_p_fa3_raw_64x128x224(float* acc_s,
                                                               FP8T* p_frag) {
  fp8_acc_to_fa3_p_regs_64x224_no_cute(acc_s, reinterpret_cast<uint32_t*>(p_frag));
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pack_p_fa3_raw_64x128x224_to_smem(
    float* acc_s, FP8T* p_smem) {
  int const tid = static_cast<int>(threadIdx.x) & 127;
  uint32_t p_regs[28];
  fp8_acc_to_fa3_p_regs_64x224_no_cute(acc_s, p_regs);
#pragma unroll
  for (int i = 0; i < 112; ++i) {
    p_smem[tid * 112 + i] = reinterpret_cast<FP8T*>(p_regs)[i];
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pack_p_fa3_raw_64x128x224_to_ptr(
    float* acc_s, FP8T* p_ptr) {
  uint32_t p_regs[28];
  fp8_acc_to_fa3_p_regs_64x224_no_cute(acc_s, p_regs);
#pragma unroll
  for (int i = 0; i < 112; ++i) {
    p_ptr[i] = reinterpret_cast<FP8T*>(p_regs)[i];
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pack_p_logical_fa3_raw_64x128x224(
    float* p_logical, int p_row_stride, FP8T* p_frag) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, Int<224>, _128>;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapePV,
                                              GMMA::Major::K, GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  float acc_s_storage[112];
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_qk.get_slice(tid);
  Tensor tSrS_template = partition_fragment_C(tiled_mma_qk, Shape<_64, Int<224>>{});
  Tensor tSrS = make_tensor(acc_s_storage, tSrS_template.layout());
  Tensor cS = make_identity_tensor(Shape<_64, Int<224>>{});
  Tensor tScS = thr_mma.partition_C(cS);

  Tensor tSrS_rowcol = make_tensor(
      tSrS.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tSrS.layout()));
  Tensor tScS_rowcol = make_tensor(
      tScS.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tScS.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
    auto coord = tScS_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    tSrS_rowcol(m, n) = p_logical[row * p_row_stride + col];
  }
  }

  fp8_gqa_detail::permute_Cregs_fp8(tSrS);
  Tensor p_acc =
      make_tensor(tSrS.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
  Tensor p = make_tensor(reinterpret_cast<Element*>(p_frag), p_acc.layout());

#pragma unroll
  for (int i = 0; i < size(p_acc); ++i) {
    fp8_e4_t packed;
    (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
        __nv_cvt_float_to_fp8(p_acc(i), __NV_SATFINITE, __NV_E4M3);
    p(i) = reinterpret_cast<Element const&>(packed);
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pack_p_fa3_raw_64x128x224_to_local_p_experimental(
    float* acc_s, FP8T* p_ptr) {
  fp8_pack_p_fa3_raw_64x128x224_to_ptr(acc_s, p_ptr);
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224(
    FP8T* p_frag, FP8T* v_tc_smem, int /*flags*/, float* delta) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x224<FP8T>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  uint32_t* p_regs = reinterpret_cast<uint32_t*>(p_frag);
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    delta[i] = 0.0f;
  }

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 7; ++ki) {
    cute::GmmaDescriptor desc_b = tOrV(_, _, ki)(0);
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b),
        reinterpret_cast<uint32_t*>(delta),
        ki != 0);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_ptx_unit_begin_from_p_smem_fa3_raw_64x128x224(
    FP8T* p_smem, FP8T* v_tc_smem, int flags, float* delta) {
  using Element = cutlass::float_e4m3_t;
  int const tid = static_cast<int>(threadIdx.x) & 127;
  Element p_frag[112];
#pragma unroll
  for (int i = 0; i < 112; ++i) {
    reinterpret_cast<FP8T*>(p_frag)[i] = p_smem[tid * 112 + i];
  }
  fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224(
      reinterpret_cast<FP8T*>(p_frag), v_tc_smem, flags, delta);
}

template <typename FP8T>
__device__ __forceinline__ void fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224(
    FP8T* p_ptr, FP8T* v_tc_smem, int flags, float* delta) {
  fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224(
      p_ptr, v_tc_smem, flags, delta);
}

template <typename FP8T>
__device__ __forceinline__
void fp8_pv_ptx_unit_begin_accumulate_from_p_smem_fa3_raw_64x128x224(
    FP8T* p_smem, FP8T* v_tc_smem, int /*flags*/, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x224<FP8T>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  Element p_frag[112];
#pragma unroll
  for (int i = 0; i < 112; ++i) {
    reinterpret_cast<FP8T*>(p_frag)[i] = p_smem[tid * 112 + i];
  }
  uint32_t* p_regs = reinterpret_cast<uint32_t*>(p_frag);

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 7; ++ki) {
    cute::GmmaDescriptor desc_b = tOrV(_, _, ki)(0);
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b),
        reinterpret_cast<uint32_t*>(acc_o),
        true);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <typename FP8T>
__device__ __forceinline__
void fp8_pv_ptx_unit_begin_accumulate_from_p_ptr_fa3_raw_64x128x224(
    FP8T* p_ptr, FP8T* v_tc_smem, int /*flags*/, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x224<FP8T>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  Element p_frag[112];
#pragma unroll
  for (int i = 0; i < 112; ++i) {
    reinterpret_cast<FP8T*>(p_frag)[i] = p_ptr[i];
  }
  uint32_t* p_regs = reinterpret_cast<uint32_t*>(p_frag);

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 7; ++ki) {
    cute::GmmaDescriptor desc_b = tOrV(_, _, ki)(0);
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b),
        reinterpret_cast<uint32_t*>(acc_o),
        true);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <typename FP8T>
__device__ __forceinline__
void fp8_pv_ptx_unit_begin_accumulate_from_local_p_experimental_64x128x224(
    FP8T* p_ptr, FP8T* v_tc_smem, int flags, float* acc_o) {
  fp8_pv_ptx_unit_begin_accumulate_from_p_ptr_fa3_raw_64x128x224(
      p_ptr, v_tc_smem, flags, acc_o);
}

template <typename FP8T>
__device__ __forceinline__
void fp8_pv_ptx_unit_begin_accumulate_from_p_frag_fa3_raw_64x128x224(
    FP8T* p_frag, FP8T* v_tc_smem, int /*flags*/, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x224<FP8T>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  uint32_t* p_regs = reinterpret_cast<uint32_t*>(p_frag);

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 7; ++ki) {
    cute::GmmaDescriptor desc_b = tOrV(_, _, ki)(0);
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b),
        reinterpret_cast<uint32_t*>(acc_o),
        true);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void fp8_wgmma_wait0_fence() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void fp8_fa3_raw_acc_rescale_64x128(float* acc_o,
                                                                float* ss) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);
  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tAccO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tAccO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    tAccO_rowcol(m, n) *= ss[row];
  }
  }
}

__device__ __forceinline__ void fp8_fa3_raw_acc_scale_64x128(float* acc_o,
                                                              float scale) {
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_o[i] *= scale;
  }
}

__device__ __forceinline__ void fp8_pv_ptx_unit_wait_update_rescale_fa3_raw_64x128(
    float* delta, int flags, float* ss, float v_scale, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tDelta = make_tensor(delta, tOrO_template.layout());
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tDelta);
  }

  Tensor tDelta_rowcol = make_tensor(
      tDelta.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tDelta.layout()));
  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tDelta_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tDelta_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    tAccO_rowcol(m, n) =
        (tAccO_rowcol(m, n) + tDelta_rowcol(m, n) * v_scale) * ss[row];
  }
  }
}

__device__ __forceinline__ void fp8_pv_ptx_unit_wait_update_fa3_raw_64x128(
    float* delta, int flags, float* ss, float v_scale, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tDelta = make_tensor(delta, tOrO_template.layout());
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tDelta);
  }

  Tensor tDelta_rowcol = make_tensor(
      tDelta.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tDelta.layout()));
  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tDelta_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tDelta_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    tAccO_rowcol(m, n) =
        tAccO_rowcol(m, n) * ss[row] + tDelta_rowcol(m, n) * v_scale;
  }
  }
}

__device__ __forceinline__ void fp8_pv_ptx_unit_wait_update_fa3_raw_64x128x224(
    float* delta, int flags, float* ss, float v_scale, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tDelta = make_tensor(delta, tOrO_template.layout());
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tDelta);
  }

  Tensor tDelta_rowcol = make_tensor(
      tDelta.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tDelta.layout()));
  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tDelta_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tDelta_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    tAccO_rowcol(m, n) =
        tAccO_rowcol(m, n) * ss[row] + tDelta_rowcol(m, n) * v_scale;
  }
  }
}

__device__ __forceinline__ void fp8_pv_ptx_unit_wait_update_tail_fa3_raw_64x128(
    float* delta, int flags, float v_scale, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tDelta = make_tensor(delta, tOrO_template.layout());
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tDelta);
  }

  Tensor tDelta_rowcol = make_tensor(
      tDelta.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tDelta.layout()));
  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tDelta_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tDelta_rowcol); ++n) {
    tAccO_rowcol(m, n) += tDelta_rowcol(m, n) * v_scale;
  }
  }
}

__device__ __forceinline__ void fp8_pv_ptx_unit_wait_update_tail_fa3_raw_64x128x224(
    float* delta, int flags, float v_scale, float* acc_o) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tDelta = make_tensor(delta, tOrO_template.layout());
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  if ((flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8(tDelta);
  }

  Tensor tDelta_rowcol = make_tensor(
      tDelta.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tDelta.layout()));
  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tDelta_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tDelta_rowcol); ++n) {
    tAccO_rowcol(m, n) += tDelta_rowcol(m, n) * v_scale;
  }
  }
}

__device__ __forceinline__ void fp8_zero_raw_acc_64(float* acc) {
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc[i] = 0.0f;
  }
}

__device__ __forceinline__ void fp8_fa3_raw_acc_store_64x128(
    float* acc_o, float* ls, int flags, float* o_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tAccO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tAccO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    int store_col = col;
    if ((flags & 4) != 0 && (flags & 2) == 0) {
      int const base = col & ~15;
      int const within = col & 15;
      int const pair = within >> 1;
      int const low = within & 1;
      int const mapped_pair = ((pair & 3) << 1) | (pair >> 2);
      store_col = base + (mapped_pair << 1) + low;
    }
    o_smem[row * 128 + store_col] = tAccO_rowcol(m, n) / ls[row];
  }
  }
}

template <typename OutT>
__device__ __forceinline__ void fp8_fa3_raw_acc_store_global_64x128(
    float* acc_o, float* ls, int flags, OutT* output, int output_row_stride) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tAccO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tAccO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    int store_col = col;
    if ((flags & 4) != 0 && (flags & 2) == 0) {
      int const base = col & ~15;
      int const within = col & 15;
      int const pair = within >> 1;
      int const low = within & 1;
      int const mapped_pair = ((pair & 3) << 1) | (pair >> 2);
      store_col = base + (mapped_pair << 1) + low;
    }
    output[row * output_row_stride + store_col] =
        static_cast<OutT>(tAccO_rowcol(m, n) / ls[row]);
  }
  }
}

template <typename OutT>
struct FP8Fa3OutputStore64x128 {
  static_assert(sizeof(OutT) == 2, "FA3-style output store expects fp16/bf16 output.");

  using SmemLayoutAtomO = decltype(cute::composition(
      cute::Swizzle<3, 3, 3>{},
      cute::Layout<cute::Shape<cute::_8, cute::_64>,
                   cute::Stride<cute::_64, cute::_1>>{}));
  using SmemLayoutO = decltype(cute::tile_to_shape(
      SmemLayoutAtomO{},
      cute::Shape<cute::_64, cute::_128>{}));
};

template <typename OutT>
__device__ __forceinline__ void fp8_fa3_raw_acc_store_smem_cute_64x128(
    float* acc_o, float* ls, int flags, OutT* o_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using StoreConfig = FP8Fa3OutputStore64x128<OutT>;
  using SmemCopyAtomO = Copy_Atom<SM90_U32x4_STSM_N, OutT>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor tOrO_template = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tAccO = make_tensor(acc_o, tOrO_template.layout());
  Tensor tOut = make_tensor_like<OutT>(tAccO);
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  Tensor tAccO_rowcol = make_tensor(
      tAccO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tAccO.layout()));
  Tensor tOut_rowcol = make_tensor(
      tOut.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOut.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tAccO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tAccO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    tOut_rowcol(m, n) = static_cast<OutT>(tAccO_rowcol(m, n) / ls[row]);
  }
  }

  if ((flags & 4) != 0 && (flags & 2) == 0) {
    fp8_gqa_detail::permute_output_fp8_Vcolmajor(tOut);
  }

  Tensor sO = make_tensor(make_smem_ptr(o_smem), typename StoreConfig::SmemLayoutO{});
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma_pv);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tid);
  Tensor taccOrO = smem_thr_copy_O.retile_S(tOut);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
}

template <typename ScratchT, typename OutT>
__device__ __forceinline__ void fp8_fa3_raw_acc_store_smem_cute_reuse_64x128(
    float* acc_o, float* ls, int flags, ScratchT* o_smem, OutT* /*output*/) {
  fp8_fa3_raw_acc_store_smem_cute_64x128(
      acc_o, ls, flags, reinterpret_cast<OutT*>(o_smem));
}

template <typename OutT>
__device__ __forceinline__ void fp8_fa3_o_smem_store_global_cute_64x128(
    OutT* o_smem, OutT* output, int output_row_stride) {
  using namespace cute;
  using StoreConfig = FP8Fa3OutputStore64x128<OutT>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  int const row_lane = tid >> 3;
  int const col_lane = tid & 7;
  typename StoreConfig::SmemLayoutO smem_layout;

#pragma unroll
  for (int row = row_lane; row < 64; row += 16) {
#pragma unroll
    for (int col_block = 0; col_block < 128; col_block += 64) {
      int const col = col_block + col_lane * 8;
      uint4 vec;
      OutT* vec_out = reinterpret_cast<OutT*>(&vec);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        vec_out[i] = o_smem[int(smem_layout(make_coord(row, col + i)))];
      }
      *reinterpret_cast<uint4*>(output + row * output_row_stride + col) = vec;
    }
  }
}

template <typename ScratchT, typename OutT>
__device__ __forceinline__ void fp8_fa3_o_smem_store_global_cute_reuse_64x128(
    ScratchT* o_smem, OutT* output, int output_row_stride) {
  fp8_fa3_o_smem_store_global_cute_64x128(
      reinterpret_cast<OutT*>(o_smem), output, output_row_stride);
}

template <typename FP8T>
__device__ __forceinline__ void fp8_dump_fa3_v_desc_64x128(FP8T* v_tc_smem,
                                                           float* out) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  if (tid != 0) {
    return;
  }

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});

  tl::GmmaDescriptor desc_tl;
  initialize_wgmma_descriptor<1, 1, 64>(desc_tl, v_tc_smem);

#pragma unroll
  for (int ki = 0; ki < size<2>(tOrV); ++ki) {
    Tensor tOrVKi = tOrV(_, _, ki);
    cute::GmmaDescriptor desc_fa3 = tOrVKi(0);
    int const base = ki * 12;
    out[base + 0] = static_cast<float>(desc_fa3.bitfield.start_address_);
    out[base + 1] = static_cast<float>(desc_fa3.bitfield.leading_byte_offset_);
    out[base + 2] = static_cast<float>(desc_fa3.bitfield.stride_byte_offset_);
    out[base + 3] = static_cast<float>(desc_fa3.bitfield.base_offset_);
    out[base + 4] = static_cast<float>(desc_fa3.bitfield.layout_type_);
    out[base + 5] = static_cast<float>(size(tOrVKi));

    tl::GmmaDescriptor desc_tl_ki = desc_tl + ((ki * 32) >> 4);
    out[base + 6] = static_cast<float>(desc_tl_ki.bitfield.start_address_);
    out[base + 7] = static_cast<float>(desc_tl_ki.bitfield.leading_byte_offset_);
    out[base + 8] = static_cast<float>(desc_tl_ki.bitfield.stride_byte_offset_);
    out[base + 9] = static_cast<float>(desc_tl_ki.bitfield.base_offset_);
    out[base + 10] = static_cast<float>(desc_tl_ki.bitfield.layout_type_);
    out[base + 11] = 1.0f;
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_dump_fa3_v_desc_64x128x224(FP8T* v_tc_smem,
                                                               float* out) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, Int<224>>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x224<FP8T>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  if (tid != 0) {
    return;
  }

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});

  tl::GmmaDescriptor desc_tl;
  initialize_wgmma_descriptor<1, 1, 64>(desc_tl, v_tc_smem);

#pragma unroll
  for (int ki = 0; ki < size<2>(tOrV); ++ki) {
    Tensor tOrVKi = tOrV(_, _, ki);
    cute::GmmaDescriptor desc_fa3 = tOrVKi(0);
    int const base = ki * 12;
    out[base + 0] = static_cast<float>(desc_fa3.bitfield.start_address_);
    out[base + 1] = static_cast<float>(desc_fa3.bitfield.leading_byte_offset_);
    out[base + 2] = static_cast<float>(desc_fa3.bitfield.stride_byte_offset_);
    out[base + 3] = static_cast<float>(desc_fa3.bitfield.base_offset_);
    out[base + 4] = static_cast<float>(desc_fa3.bitfield.layout_type_);
    out[base + 5] = static_cast<float>(size(tOrVKi));

    tl::GmmaDescriptor desc_tl_ki = desc_tl + ((ki * 32) >> 4);
    out[base + 6] = static_cast<float>(desc_tl_ki.bitfield.start_address_);
    out[base + 7] = static_cast<float>(desc_tl_ki.bitfield.leading_byte_offset_);
    out[base + 8] = static_cast<float>(desc_tl_ki.bitfield.stride_byte_offset_);
    out[base + 9] = static_cast<float>(desc_tl_ki.bitfield.base_offset_);
    out[base + 10] = static_cast<float>(desc_tl_ki.bitfield.layout_type_);
    out[base + 11] = 1.0f;
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_dump_fa3_pv_cute_vs_ptx_raw_o_64x128x128(
    float* acc_s, FP8T* v_tc_smem, float* out) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapeQK = Shape<_64, _128, _128>;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaQK = decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeQK>());
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaQK = decltype(make_tiled_mma(MmaQK{}, AtomLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));
  using VConfig = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  int const tid = static_cast<int>(threadIdx.x) & 127;
  float acc_s_cute_storage[64];
  float* acc_s_cute = acc_s_cute_storage;
  uint32_t p_regs[16];
  float o_ptx[64];

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc_s_cute[i] = acc_s[i];
    o_ptx[i] = 0.0f;
  }

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);

  Tensor tSrS_template = partition_fragment_C(tiled_mma_qk, Shape<_64, _128>{});
  Tensor tSrS = make_tensor(acc_s_cute, tSrS_template.layout());
  fp8_gqa_detail::permute_Cregs_fp8(tSrS);
  Tensor tOrP_acc = make_tensor(
      tSrS.data(), fp8_gqa_detail::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
  Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
  Tensor sV = make_tensor(make_smem_ptr(v_tc_smem), typename VConfig::SmemLayoutVtMma{});
  Tensor tOrV = thr_mma.partition_fragment_B(sV)(_, _, _, _0{});
  Tensor tOrO = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});

#pragma unroll
  for (int i = 0; i < size(tOrP); ++i) {
    fp8_e4_t packed;
    (reinterpret_cast<__nv_fp8_storage_t*>(&packed))[0] =
        __nv_cvt_float_to_fp8(tOrP_acc(i), __NV_SATFINITE, __NV_E4M3);
    tOrP(i) = reinterpret_cast<Element const&>(packed);
  }

  clear(tOrO);
  warpgroup_fence_operand(tOrP);
  warpgroup_fence_operand(tOrO);
  warpgroup_arrive();
  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

#pragma unroll
  for (int ki = 0; ki < size<2>(tOrP); ++ki) {
    cute::gemm(tiled_mma_pv, tOrP(_, _, ki), tOrV(_, _, ki), tOrO);
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
  }

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tOrO);
  warpgroup_fence_operand(tOrP);

  tl::GmmaDescriptor desc_b;
  initialize_wgmma_descriptor<1, 1, 64>(desc_b, v_tc_smem);
  fp8_acc_to_fa3_p_regs_64x128_no_cute(acc_s, p_regs);

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    wgmma_rs<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3, DataType::kFloat32,
             64, 128, 32, false, false>(
        p_regs + ki * 4,
        uint64_t(desc_b + ((ki * 32) >> 4)),
        reinterpret_cast<uint32_t*>(o_ptx),
        ki != 0);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    out[tid * 128 + i] = tOrO(i);
    out[tid * 128 + 64 + i] = o_ptx[i];
  }
}

__device__ __forceinline__ void fp8_raw_fragment_pattern_64x128(float* frag) {
  int const tid = static_cast<int>(threadIdx.x) & 127;
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    frag[i] = static_cast<float>(tid * 64 + i);
  }
}

__device__ __forceinline__ void fp8_cute_o_raw_store_pattern_64x128(float* o_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOrO = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

#pragma unroll
  for (int i = 0; i < size(tOrO); ++i) {
    tOrO(i) = static_cast<float>(tid * 64 + i);
  }

  Tensor tOrO_rowcol = make_tensor(
      tOrO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOrO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tOrO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tOrO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    o_smem[row * 128 + col] = tOrO_rowcol(m, n);
  }
  }
  __syncthreads();
}

__device__ __forceinline__ void fp8_cute_o_raw_store_pattern_permuted_64x128(float* o_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  int const tid = static_cast<int>(threadIdx.x) & 127;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(tid);
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOrO = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);

#pragma unroll
  for (int i = 0; i < size(tOrO); ++i) {
    tOrO(i) = static_cast<float>(tid * 64 + i);
  }
  fp8_gqa_detail::permute_output_fp8(tOrO);

  Tensor tOrO_rowcol = make_tensor(
      tOrO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOrO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tOrO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tOrO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    o_smem[row * 128 + col] = tOrO_rowcol(m, n);
  }
  }
  __syncthreads();
}

__device__ __forceinline__ void fp8_cute_o_store_pattern_64x128(float* o_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(static_cast<int>(threadIdx.x) & 127);
  Tensor cO = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOrO = partition_fragment_C(tiled_mma_pv, Shape<_64, _128>{});
  Tensor tOcO = thr_mma.partition_C(cO);
  Tensor tOrO_rowcol = make_tensor(
      tOrO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOrO.layout()));
  Tensor tOcO_rowcol = make_tensor(
      tOcO.data(), fp8_gqa_detail::convert_layout_acc_rowcol(tOcO.layout()));

#pragma unroll
  for (int m = 0; m < size<0>(tOrO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tOrO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    tOrO_rowcol(m, n) = float(row * 128 + col);
  }
  }

#pragma unroll
  for (int m = 0; m < size<0>(tOrO_rowcol); ++m) {
#pragma unroll
  for (int n = 0; n < size<1>(tOrO_rowcol); ++n) {
    auto coord = tOcO_rowcol(m, n);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    o_smem[row * 128 + col] = tOrO_rowcol(m, n);
  }
  }
  __syncthreads();
}

template <typename FP8T>
__device__ __forceinline__ void fp8_cute_v_direct_roundtrip_128x128(
    FP8T* v_smem, FP8T* v_tc_smem, FP8T* out) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));
  int const tid = static_cast<int>(threadIdx.x) & 127;

#pragma unroll
  for (int d = 0; d < 128; ++d) {
    int const n = tid;
    sV(d, n, 0) = v_smem[n * 128 + d];
  }
  __syncthreads();

#pragma unroll
  for (int d = 0; d < 128; ++d) {
    int const n = tid;
    out[d * 128 + n] = sV(d, n, 0);
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_cute_v_fa3_roundtrip_128x128(
    FP8T* v_smem, FP8T* v_vt_smem, FP8T* v_tc_smem, FP8T* out) {
  using namespace cute;
  using Config = fp8_gqa_detail::VTranspose128x128Fa3Src<FP8T>;

  fp8_pack_v_128x128_fa3_vt(v_smem, v_vt_smem);
  __syncthreads();
  fp8_transpose_v_128x128_fa3_src_ldsm_stsm(v_vt_smem, v_tc_smem);
  __syncthreads();

  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_tc_smem), typename Config::SmemLayoutVtMma{}));
  int const tid = static_cast<int>(threadIdx.x) & 127;

#pragma unroll
  for (int d = 0; d < 128; ++d) {
    int const n = tid;
    out[d * 128 + n] = sV(d, n, 0);
  }
}

template <typename FP8T>
__device__ __forceinline__ void fp8_cute_p_a_roundtrip_64x128(
    FP8T* p_smem, FP8T* p_out_smem) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(static_cast<int>(threadIdx.x) & 127);
  Tensor sP = make_tensor(make_smem_ptr(p_smem), Layout<Shape<_64, _128>, Stride<_128, _1>>{});
  Tensor sPOut = make_tensor(make_smem_ptr(p_out_smem), Layout<Shape<_64, _128>, Stride<_128, _1>>{});
  Tensor tOsP = thr_mma.partition_A(sP);
  Tensor tOsPOut = thr_mma.partition_A(sPOut);
  Tensor tOrP = thr_mma.make_fragment_A(tOsP);

  cute::copy(tOsP, tOrP);
  cute::copy(tOrP, tOsPOut);
  __syncthreads();
}

template <typename FP8T>
__device__ __forceinline__ void fp8_cute_p_a_dump_64x128(FP8T* p_smem, FP8T* out) {
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using TileShapePV = Shape<_64, _128, _128>;
  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using MmaPV = decltype(GMMA::rs_op_selector<Element, Element, ElementAccum,
                                              TileShapePV, GMMA::Major::K,
                                              GMMA::Major::K>());
  using TiledMmaPV = decltype(make_tiled_mma(MmaPV{}, AtomLayout{}));

  TiledMmaPV tiled_mma_pv;
  auto thr_mma = tiled_mma_pv.get_slice(static_cast<int>(threadIdx.x) & 127);
  Tensor sP = make_tensor(make_smem_ptr(p_smem), Layout<Shape<_64, _128>, Stride<_128, _1>>{});
  Tensor cP = make_identity_tensor(Shape<_64, _128>{});
  Tensor tOsP = thr_mma.partition_A(sP);
  Tensor tOcP = thr_mma.partition_A(cP);
  Tensor tOrP = thr_mma.make_fragment_A(tOsP);

  cute::copy(tOsP, tOrP);

#pragma unroll
  for (int i = 0; i < size(tOrP); ++i) {
    auto coord = tOcP(i);
    int const row = int(get<0>(coord));
    int const col = int(get<1>(coord));
    out[row * 128 + col] = reinterpret_cast<FP8T const&>(tOrP(i));
  }
}

}  // namespace tl
