#pragma once

#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

#include "warp_uniform.h"

namespace tileops {

// WGMMA descriptor of a shared tile at p under swizzle mode Layout (1:
// 128-byte, 3: 32-byte), with its leading and stride byte offsets in 16-byte
// units. Key steps build it before the warpgroup barrier, so the products issue
// right after the barrier.
template <int Layout, int LeadingByteOffset, int StrideByteOffset, typename T>
__device__ __forceinline__ uint64_t attn_desc_swizzled(T* p) {
  tl::GmmaDescriptor d;
  tl::initialize_wgmma_descriptor<Layout, LeadingByteOffset, StrideByteOffset>(
      d, p);
  // An opaque copy: ptxas keeps the value instead of rebuilding it from the
  // shared window base inside the key loop, after the warpgroup barrier.
  uint64_t v;
  asm volatile("mov.b64 %0, %1;" : "=l"(v) : "l"(d.desc_));
  return v;
}

// The descriptor of a 128-byte-swizzled tile: 8 rows 1024 bytes apart.
template <int LeadingByteOffset, typename T>
__device__ __forceinline__ uint64_t attn_desc(T* p) {
  return attn_desc_swizzled<1, LeadingByteOffset, 64>(p);
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

// A head of one 64-channel block under the 128-byte swizzle and an Edge-channel
// edge (0 or 16) under the 32-byte swizzle, the tiles `make_swizzled_layout`
// gives them. The descriptors are built once an item; a step adds its stage.

// acc[64 x 128] = q[64 x (64 + Edge)] . k[128 x (64 + Edge)]^T.
template <tl::DataType DT, int Edge>
__device__ __forceinline__ void attn_qk_64x128_w64(float* acc, uint64_t q,
                                                   uint64_t k, uint64_t qe,
                                                   uint64_t ke) {
  static_assert(Edge == 0 || Edge == 16, "an edge of one 16-channel K step");
  tl::warpgroup_fence_operand(acc, 64);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    tl::wgmma_ss<DT, DT, tl::DataType::kFloat32, 64, 128, 16, false, false, 1,
                 1>(q + uint64_t(ki * 2), k + uint64_t(ki * 2),
                    reinterpret_cast<uint32_t*>(acc), ki > 0 ? 1 : 0);
  }
  if constexpr (Edge > 0) {
    tl::wgmma_ss<DT, DT, tl::DataType::kFloat32, 64, 128, 16, false, false, 1,
                 1>(qe, ke, reinterpret_cast<uint32_t*>(acc), 1);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(acc, 64);
}

// acc_o[64 x 64] += p[64 x 128] . v[128 x 64] and acc_e[64 x Edge] += p . ve,
// with p in shared memory (two 64-column blocks under the 128-byte swizzle) and
// v, ve MN-major.
template <tl::DataType DT, int Edge>
__device__ __forceinline__ void attn_pv_64x128_w64(float* acc_o, float* acc_e,
                                                   uint64_t p, uint64_t v,
                                                   uint64_t ve) {
  static_assert(Edge == 0 || Edge == 16, "an edge of one 16-column N block");
  tl::warpgroup_fence_operand(acc_o, 32);
  if constexpr (Edge > 0) tl::warpgroup_fence_operand(acc_e, Edge / 2);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 8; ++ki) {
    tl::wgmma_ss<DT, DT, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
        p + uint64_t(((ki >> 2) * 8192 + (ki & 3) * 32) >> 4),
        v + uint64_t(ki * 128), reinterpret_cast<uint32_t*>(acc_o), 1);
  }
  if constexpr (Edge > 0) {
#pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
      tl::wgmma_ss<DT, DT, tl::DataType::kFloat32, 64, Edge, 16, false, true, 1,
                   1>(p + uint64_t(((ki >> 2) * 8192 + (ki & 3) * 32) >> 4),
                      ve + uint64_t(ki * 32),
                      reinterpret_cast<uint32_t*>(acc_e), 1);
    }
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(acc_o, 32);
  if constexpr (Edge > 0) tl::warpgroup_fence_operand(acc_e, Edge / 2);
}

// Store the warpgroup's 64 x 128 probabilities, held in the m64n128
// accumulator's layout, to p in 16 bits: two 64-column blocks under the
// 128-byte swizzle, one stmatrix a pair of 8-column groups. A warp's WGMMA
// reads only its own 16 rows of p, so the proxy fence and a warp barrier hand
// them to it.
template <tl::DataType DT, typename T>
__device__ __forceinline__ void attn_store_p_64x128(T* p, const float* acc) {
  const int lane = threadIdx.x & 31;
  const int matrix = lane >> 3;
  const int row = ((threadIdx.x >> 5) & 3) * 16 + (matrix & 1) * 8 + (lane & 7);
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(p));
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    const int col = (j + (matrix >> 1)) * 8;
    const uint32_t addr = base + (col >> 6) * 8192 + row * 128 +
                          ((((col & 63) >> 3) ^ (row & 7)) << 4);
    uint32_t r[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const float* a = acc + (j + (h >> 1)) * 4 + (h & 1) * 2;
      if constexpr (DT == tl::DataType::kBFloat16) {
        __nv_bfloat162 v = __floats2bfloat162_rn(a[0], a[1]);
        r[h] = reinterpret_cast<uint32_t&>(v);
      } else {
        __half2 v = __floats2half2_rn(a[0], a[1]);
        r[h] = reinterpret_cast<uint32_t&>(v);
      }
    }
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::
            "r"(addr),
        "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]));
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::);
  __syncwarp();
}

}  // namespace tileops
