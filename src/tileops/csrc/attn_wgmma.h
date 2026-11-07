#pragma once

#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

#include "warp_uniform.h"

namespace tileops {

// WGMMA descriptor of a 128-byte-swizzled shared tile at p. Key steps build it
// before the warpgroup barrier, so the products issue right after the barrier.
template <int LeadingByteOffset, typename T>
__device__ __forceinline__ uint64_t attn_desc(T* p) {
  tl::GmmaDescriptor d;
  tl::initialize_wgmma_descriptor<1, LeadingByteOffset, 64>(d, p);
  // An opaque copy: ptxas keeps the value instead of rebuilding it from the
  // shared window base inside the key loop, after the warpgroup barrier.
  uint64_t v;
  asm volatile("mov.b64 %0, %1;" : "=l"(v) : "l"(d.desc_));
  return v;
}

// Shared tiles use the 128-byte swizzle `make_swizzled_layout` gives 16-bit
// rows of 128 channels: two 64-channel column blocks, `rows * 128` bytes apart.

// acc[64 x 128] = q[64 x 128] . k[128 x 128]^T, q and k given by descriptors.
template <tl::DataType DT>
__device__ __forceinline__ void attn_qk_64x128_d128(float* acc, uint64_t q,
                                                    uint64_t k) {
  tl::warpgroup_fence_operand(acc, 64);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 8; ++ki) {
    tl::wgmma_ss<DT, DT, tl::DataType::kFloat32, 64, 128, 16, false, false, 1,
                 1>(q + uint64_t((((ki >> 2) * 8192 + (ki & 3) * 32) >> 4)),
                    k + uint64_t((((ki >> 2) * 16384 + (ki & 3) * 32) >> 4)),
                    reinterpret_cast<uint32_t*>(acc), ki > 0 ? 1 : 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(acc, 64);
}

// acc[64 x 128] += p[64 x 128] . v[128 x 128], p in registers as the QK
// accumulator's layout cast to 16 bits, v given by its descriptor.
template <tl::DataType DT, typename P>
__device__ __forceinline__ void attn_pv_64x128_d128(float* acc, P* p,
                                                    uint64_t v) {
  tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(p), 32);
  tl::warpgroup_fence_operand(acc, 64);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 8; ++ki) {
    tl::wgmma_rs<DT, DT, tl::DataType::kFloat32, 64, 128, 16, false, true, 1,
                 1>(reinterpret_cast<const uint32_t*>(p + ki * 8),
                    v + uint64_t((ki * 2048) >> 4),
                    reinterpret_cast<uint32_t*>(acc), 1);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(acc, 64);
  tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(p), 32);
}

}  // namespace tileops
