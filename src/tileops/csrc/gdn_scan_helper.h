#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

namespace tileops {

// The state warp groups of the gated delta rule's one-kernel prefill. Each
// holds S^T, 64 value rows, reads the chunk's 64 x 128 value tile V and writes
// its output O transposed. Both tiles are 64 tokens of 128 values in two
// 64-column atoms of the 128-byte swizzle, the second atom 8 KB on: the layout
// make_wgmma_swizzled_layout gives a 64-element 16-bit atom.
//
// A 64 x 64 float32 wgmma accumulator puts rows 16 w + l / 4 and 16 w + l / 4 +
// 8 in lane l of warp w, and of each eight-column block j the columns 8 j + 2
// (l % 4) and the next in registers 4 j, 4 j + 1 (upper row) and 4 j + 2, 4 j +
// 3 (lower row). Packed in pairs, the same registers are the A operand of a k16
// product: k-step i is the pairs 4 i to 4 i + 3.

constexpr int kGdnScanGroupThreads = 128;
constexpr int kGdnScanTokens = 64;
constexpr int kGdnScanRowBytes = 128;
constexpr int kGdnScanAtomBytes = kGdnScanTokens * kGdnScanRowBytes;

// Arrive at named barrier id, counted over n threads, without waiting.
__device__ __forceinline__ void gdn_bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}

// Byte offset of (token, value) in the two-atom tile.
__device__ __forceinline__ int gdn_tile_offset(int token, int value) {
  return (value >> 6) * kGdnScanAtomBytes + token * kGdnScanRowBytes +
         (((((value & 63) >> 3) ^ (token & 7)) << 4) | ((value & 7) << 1));
}

__device__ __forceinline__ float2 gdn_unpack(uint32_t x, bfloat16_t*) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&x));
}

__device__ __forceinline__ float2 gdn_unpack(uint32_t x, half_t*) {
  return __half22float2(*reinterpret_cast<__half2*>(&x));
}

__device__ __forceinline__ uint32_t gdn_pack(float a, float b, bfloat16_t*) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t gdn_pack(float a, float b, half_t*) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// X^T = V^T - SK^T diag(e^gc) as the A operand of the product X^T T^T, for
// value rows row0 to row0 + 63: V^T's 8 x 8 tiles read transposed by ldmatrix
// straight into the operand's registers. sk is S^T K^T's accumulator; eg the
// chunk's 64 e^gc. T carries the step sizes.
template <typename InT>
__device__ __forceinline__ void gdn_form_x(InT* x, const float* sk,
                                           const InT* v, int row0,
                                           const float* eg, int tx) {
  int warp = (tx % kGdnScanGroupThreads) >> 5;
  int lane = tx & 31;
  int quad = lane >> 3;
  int value = row0 + warp * 16 + (quad & 1) * 8;
  const char* base = reinterpret_cast<const char*>(v);
  uint32_t* xr = reinterpret_cast<uint32_t*>(x);
  int c = (lane & 3) << 1;
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    int token = ki * 16 + (quad >> 1) * 8 + (lane & 7);
    uint32_t addr = smem_ptr_to_uint(base + gdn_tile_offset(token, value));
    uint32_t r[4];
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, "
        "[%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      // Pair p: rows upper, lower, upper, lower; tokens 16 ki + 2 (l % 4), 8 on
      // for p >= 2.
      int col = ki * 16 + c + (p >> 1) * 8;
      float2 vv = gdn_unpack(r[p], x);
      float2 e = *reinterpret_cast<const float2*>(eg + col);
      xr[4 * ki + p] = gdn_pack(vv.x - sk[8 * ki + 2 * p] * e.x,
                                vv.y - sk[8 * ki + 2 * p + 1] * e.y, x);
    }
  }
}

// Issue acc = X T^T without waiting: X^T's 64 x 64 A operand in registers, T a
// 64 x 64 tile in the 128-byte swizzle. The auxiliary group fences T into the
// async proxy before it signals T ready, so no fence precedes the product.
template <tl::DataType DT, typename InT>
__device__ __forceinline__ void gdn_issue_xt(float* acc, InT* x, InT* t) {
  static_assert(DT == tl::DataType::kBFloat16 || DT == tl::DataType::kFloat16,
                "gdn_issue_xt takes bfloat16 or float16 operands");
  tl::GmmaDescriptor desc;
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc, t);
  uint32_t* a = reinterpret_cast<uint32_t*>(x);
  tl::warpgroup_fence_operand(a, 16);
  tl::warpgroup_fence_operand(acc, 32);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    tl::wgmma_rs<DT, DT, tl::DataType::kFloat32, 64, 64, 16, false, false, 1,
                 1>(a + 4 * ki, uint64_t(desc + ((ki * 32) >> 4)),
                    reinterpret_cast<uint32_t*>(acc), ki > 0);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(acc, 32);
  tl::warpgroup_fence_operand(a, 16);
}

// O^T's accumulator (value rows row0 to row0 + 63, tokens as columns) into the
// output tile transposed: two eight-token blocks of a warp's 16 value rows a
// stmatrix.
template <typename OutT>
__device__ __forceinline__ void gdn_store_ot(const float* o, OutT* out,
                                             int row0, int tx) {
  int warp = (tx % kGdnScanGroupThreads) >> 5;
  int lane = tx & 31;
  int quad = lane >> 3;
  char* base = reinterpret_cast<char*>(out);
  int value = row0 + warp * 16 + (quad & 1) * 8;
#pragma unroll
  for (int j = 0; j < 8; j += 2) {
    int token = (j + (quad >> 1)) * 8 + (lane & 7);
    uint32_t addr = smem_ptr_to_uint(base + gdn_tile_offset(token, value));
    uint32_t r0 = gdn_pack(o[4 * j], o[4 * j + 1], out);
    uint32_t r1 = gdn_pack(o[4 * j + 2], o[4 * j + 3], out);
    uint32_t r2 = gdn_pack(o[4 * j + 4], o[4 * j + 5], out);
    uint32_t r3 = gdn_pack(o[4 * j + 6], o[4 * j + 7], out);
    asm volatile(
        "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, "
        "%4};\n" ::"r"(addr),
        "r"(r0), "r"(r1), "r"(r2), "r"(r3));
  }
}

// Tokens 0 to valid - 1 of atom `half` of the output tile to global memory,
// token r at out + r * row_stride, by one warp in 16-byte pieces. Tokens from
// 64 on are never written.
template <typename OutT>
__device__ __forceinline__ void gdn_copy_rows(const OutT* tile, int half,
                                              OutT* out, int row_stride,
                                              int valid, int lane) {
  constexpr int kPieces = kGdnScanRowBytes / 16;
  const char* base =
      reinterpret_cast<const char*>(tile) + half * kGdnScanAtomBytes;
  for (int i = lane; i < kGdnScanTokens * kPieces; i += 32) {
    int r = i / kPieces;
    int c = i % kPieces;
    if (r < valid) {
      uint4 x = *reinterpret_cast<const uint4*>(base + r * kGdnScanRowBytes +
                                                ((c ^ (r & 7)) << 4));
      *reinterpret_cast<uint4*>(out + r * row_stride +
                                c * (16 / sizeof(OutT))) = x;
    }
  }
}

__device__ __forceinline__ void gdn_form_x_bf16(bfloat16_t* x, float* sk,
                                                bfloat16_t* v, int row0,
                                                float* eg, int tx) {
  gdn_form_x(x, sk, v, row0, eg, tx);
}

__device__ __forceinline__ void gdn_form_x_fp16(half_t* x, float* sk, half_t* v,
                                                int row0, float* eg, int tx) {
  gdn_form_x(x, sk, v, row0, eg, tx);
}

__device__ __forceinline__ void gdn_issue_xt_bf16(float* acc, bfloat16_t* x,
                                                  bfloat16_t* t) {
  gdn_issue_xt<tl::DataType::kBFloat16>(acc, x, t);
}

__device__ __forceinline__ void gdn_issue_xt_fp16(float* acc, half_t* x,
                                                  half_t* t) {
  gdn_issue_xt<tl::DataType::kFloat16>(acc, x, t);
}

__device__ __forceinline__ void gdn_store_ot_bf16(float* o, bfloat16_t* out,
                                                  int row0, int tx) {
  gdn_store_ot(o, out, row0, tx);
}

__device__ __forceinline__ void gdn_store_ot_fp16(float* o, half_t* out,
                                                  int row0, int tx) {
  gdn_store_ot(o, out, row0, tx);
}

__device__ __forceinline__ void gdn_copy_rows_bf16(bfloat16_t* tile, int half,
                                                   bfloat16_t* out,
                                                   int row_stride, int valid,
                                                   int lane) {
  gdn_copy_rows(tile, half, out, row_stride, valid, lane);
}

__device__ __forceinline__ void gdn_copy_rows_fp16(half_t* tile, int half,
                                                   half_t* out, int row_stride,
                                                   int valid, int lane) {
  gdn_copy_rows(tile, half, out, row_stride, valid, lane);
}

}  // namespace tileops
