#pragma once

#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

namespace tileops {

// One 128-wide K step of a 64 x N FP8 NT product, issued as four N-wide k32
// WGMMAs into a fresh accumulator and committed as one group.
//
// The group is left in flight: the caller waits for it and then fences the
// accumulator. A fence between the commit and the wait, as ``T.wgmma_gemm``
// and ``dense_wgmma_8bit_64xN`` emit, reads the accumulator while the WGMMAs
// run, and ptxas then serializes every k32 WGMMA of the kernel (C7514).
//
// a_smem is one stage's 64 x 128 tile and b_smem its N x 128 tile, each row one
// 128-byte swizzle atom as ``make_swizzled_layout`` lays out a 128-element fp8
// row. Register ``r`` of a thread holds row ``16 * warp + lane / 4 + 8 * ((r %
// 4) / 2)`` and column ``8 * (r / 4) + 2 * (lane % 4) + r % 2`` of the 64 x N
// tile.
template <int N>
__device__ __forceinline__ void fp8_stream_wgmma_64xN_k128(float* accumulator,
                                                           fp8_e4_t* a_smem,
                                                           fp8_e4_t* b_smem) {
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
        reinterpret_cast<uint32_t*>(accumulator), 0 < ki ? 1 : 0);
  }
  tl::warpgroup_commit_batch();
}

}  // namespace tileops
