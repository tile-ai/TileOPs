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

static constexpr int kHeadDimV = 128;
static constexpr int kBlockN = 128;
static constexpr int kBlockN224 = 224;
static constexpr int kStages = 1;
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
__device__ __forceinline__ void fp8_acc_to_fa3_p_regs_64x224_no_cute(
    float* acc_s, uint32_t* p_regs) {
#pragma unroll
  for (int r = 0; r < 28; ++r) {
    p_regs[r] = fp8_pack4_fa3_p_bytes(acc_s, r * 4);
  }
}
__device__ __forceinline__ void fp8_producer_barrier_128() {
  asm volatile("bar.sync 15, 128;\n" ::: "memory");
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
__device__ __forceinline__ void fp8_zero_raw_acc_64(float* acc) {
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc[i] = 0.0f;
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
}  // namespace tl
