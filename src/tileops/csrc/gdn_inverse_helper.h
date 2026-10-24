#pragma once

#include <cuda_fp16.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

namespace tileops {

// One warp group's per-chunk products for the gated delta rule's one-kernel
// prefill:
//
//   T = (I + Lg)^-1 diag(beta),  Lg = strict_lower(diag(beta) (G * K K^T)),
//   P = scale lower(G * Q K^T),  G[r, c] = e^(gc_r - gc_c),
//
// for a 64-token chunk with 128-wide queries and keys. Q and K are the 64 x 128
// tiles TileLang stages for wgmma: two 64-column atoms of the 128-byte swizzle,
// the second 8 KB on. T, P and the two exchange tiles are 64 x 64 16-bit tiles,
// each row 128 bytes in the 128-byte swizzle.
//
// A 64 x 64 float32 accumulator is in the wgmma layout: warp w holds rows 16 w
// to 16 w + 15; lane l holds rows 16 w + l / 4 and 16 w + l / 4 + 8, and of
// each eight-column block j the columns 8 j + 2 (l % 4) and the next, in
// registers 4 j, 4 j + 1 (upper row) and 4 j + 2, 4 j + 3 (lower row). Its 16 x
// 16 block (w, b) is registers 8 b to 8 b + 7: the same eight floats a 16 x 16
// mma.sync accumulator holds.

constexpr int kGdnChunk = 64;
constexpr int kGdnGroupThreads = 128;
constexpr int kGdnSwizzleRowBytes = 128;
// Bytes between the two 64-column atoms of a 64 x 128 16-bit tile.
constexpr int kGdnAtomBytes = kGdnChunk * kGdnSwizzleRowBytes;

// Byte offset of element (row, col) of a 64-wide 16-bit row-major tile in the
// 128-byte swizzle: 16-byte column groups permuted by the row's low three bits.
__device__ __forceinline__ int gdn_swizzle(int row, int col) {
  return row * kGdnSwizzleRowBytes +
         ((((col >> 3) ^ (row & 7)) << 4) | ((col & 7) << 1));
}

__device__ __forceinline__ uint32_t gdn_pack_half(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t gdn_pack_out(float lo, float hi, half_t*) {
  return gdn_pack_half(lo, hi);
}

__device__ __forceinline__ uint32_t gdn_pack_out(float lo, float hi,
                                                 bfloat16_t*) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ float2 gdn_unpack_half(uint32_t x) {
  return __half22float2(*reinterpret_cast<__half2*>(&x));
}

__device__ __forceinline__ uint32_t gdn_transpose8(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
               : "=r"(y)
               : "r"(x));
  return y;
}

// c (16 x 8) += a (16 x 16, A layout) b (16 x 8, B layout), float16 in, float32
// out.
__device__ __forceinline__ void gdn_mma16816(float* c, const uint32_t* a,
                                             uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A 16 x 16 block's eight accumulator floats as an A operand: four packed
// float16 pairs.
__device__ __forceinline__ void gdn_pack16(const float* f, uint32_t* a) {
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = gdn_pack_half(f[2 * i], f[2 * i + 1]);
}

// The B operand of a 16 x 16 block held as an A operand: its 8 x 8 tiles
// transposed.
__device__ __forceinline__ void gdn_as_b16(const uint32_t* a, uint32_t* b) {
#pragma unroll
  for (int i = 0; i < 4; ++i) b[i] = gdn_transpose8(a[i]);
}

// acc (16 x 16) += a b, a an A operand and b a B operand.
__device__ __forceinline__ void gdn_mma16(float* acc, const uint32_t* a,
                                          const uint32_t* b) {
  gdn_mma16816(acc, a, b[0], b[1]);
  gdn_mma16816(acc + 4, a, b[2], b[3]);
}

__device__ __forceinline__ void gdn_add_identity16(float* f, int lane) {
  int r = lane >> 2;
  int c = (lane & 3) << 1;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int row = r + ((i >> 1) & 1) * 8;
    int col = c + (i & 1) + (i >> 2) * 8;
    f[i] += row == col ? 1.0f : 0.0f;
  }
}

// The B operand of block (rb, cb) of a swizzled float16 tile, read transposed.
__device__ __forceinline__ void gdn_load_b16(const half_t* tile, int rb, int cb,
                                             uint32_t* b, int lane) {
  int m = lane >> 3;
  int row = rb * 16 + (m & 1) * 8 + (lane & 7);
  int col = cb * 16 + (m >> 1) * 8;
  uint32_t addr = smem_ptr_to_uint(reinterpret_cast<const char*>(tile) +
                                   gdn_swizzle(row, col));
  asm volatile(
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
      : "r"(addr));
}

// A 16 x 16 block held as an A operand, to block (rb, cb) of a swizzled float16
// tile.
__device__ __forceinline__ void gdn_store_a16(half_t* tile, int rb, int cb,
                                              const uint32_t* a, int lane) {
  int m = lane >> 3;
  int row = rb * 16 + (m & 1) * 8 + (lane & 7);
  int col = cb * 16 + (m >> 1) * 8;
  uint32_t addr =
      smem_ptr_to_uint(reinterpret_cast<char*>(tile) + gdn_swizzle(row, col));
  asm volatile(
      "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::
          "r"(addr),
      "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
}

// c (16 x 16) += a times block (rb, cb) of a swizzled float16 tile.
__device__ __forceinline__ void gdn_mma_tile(float* c, const uint32_t* a,
                                             const half_t* tile, int rb, int cb,
                                             int lane) {
  uint32_t b[4];
  gdn_load_b16(tile, rb, cb, b, lane);
  gdn_mma16(c, a, b);
}

// A warp group's 64 x 64 accumulator to a swizzled 16-bit tile.
template <typename OutT>
__device__ __forceinline__ void gdn_store64(const float* acc, OutT* tile,
                                            int tx) {
  int warp = (tx % kGdnGroupThreads) >> 5;
  int lane = tx & 31;
  int row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
  char* base = reinterpret_cast<char*>(tile);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint32_t r0 = gdn_pack_out(acc[4 * j], acc[4 * j + 1], tile);
    uint32_t r1 = gdn_pack_out(acc[4 * j + 2], acc[4 * j + 3], tile);
    uint32_t addr = smem_ptr_to_uint(base + gdn_swizzle(row, j * 8));
    asm volatile(
        "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(
            addr),
        "r"(r0), "r"(r1));
  }
}

// tx through an opaque move, so values derived from it are formed where they
// are used instead of hoisted out of the caller's loop and held, or spilled,
// across it.
__device__ __forceinline__ int gdn_opaque(int tx) {
  int y;
  asm volatile("mov.b32 %0, %1;\n" : "=r"(y) : "r"(tx));
  return y;
}

__device__ __forceinline__ void gdn_group_sync(int barrier_id) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(barrier_id), "n"(kGdnGroupThreads));
}

// Xd = (I + D)^-1 for a warp's 16 x 16 strictly lower block D, exact since D^16
// = 0: Xd = (I - D + D^2 - D^3)(I + D^4)(I + D^8), D^3 and D^4 both from D^2.
__device__ __forceinline__ void gdn_diag_inverse(const float* d, float* x,
                                                 int lane) {
  uint32_t ad[4], bd[4], a2[4], b2[4], a4[4], b4[4], a8[4], b8[4], ax[4];
  float p[8], q[8];
  gdn_pack16(d, ad);
  gdn_as_b16(ad, bd);
#pragma unroll
  for (int i = 0; i < 8; ++i) p[i] = q[i] = 0.0f;
  gdn_mma16(p, ad, bd);  // D^2
  gdn_pack16(p, a2);
  gdn_as_b16(a2, b2);
  gdn_mma16(q, a2, bd);  // D^3
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = p[i] - d[i] - q[i];
  gdn_add_identity16(x, lane);
#pragma unroll
  for (int i = 0; i < 8; ++i) p[i] = 0.0f;
  gdn_mma16(p, a2, b2);  // D^4
  gdn_pack16(p, a4);
  gdn_as_b16(a4, b4);
#pragma unroll
  for (int i = 0; i < 8; ++i) p[i] = 0.0f;
  gdn_mma16(p, a4, b4);  // D^8
  gdn_pack16(p, a8);
  gdn_as_b16(a8, b8);
  gdn_pack16(x, ax);
  gdn_mma16(x, ax, b4);
  gdn_pack16(x, ax);
  gdn_mma16(x, ax, b8);
}

// T = (I + L)^-1 diag(col_scale) for L strictly lower in the accumulator
// layout, each warp forming its own 16 rows after one exchange through shared
// memory.
//
// With Ld the 16 x 16 diagonal blocks of L, Lo the rest, Xd = (I + Ld)^-1 and N
// = Xd Lo: I + L = (I + Ld)(I + N), N is strictly block lower so N^4 = 0, and
// T = (I - N + N^2 - N^3) Xd. Xd is block diagonal, so warp w forms Xd_w and
// its own row blocks N_wj = Xd_w Lo_wj alone. Block (w, j) of T then needs only
// N_10, N_20, N_21 and Xd_0 to Xd_2 from the other warps:
//   T_ww = Xd_w;  T_10 = -N_10 Xd_0;
//   T_20 = (N_21 N_10 - N_20) Xd_0;  T_21 = -N_21 Xd_1;
//   T_30 = ((N_31 - N_32 N_21) N_10 + N_32 N_20 - N_30) Xd_0;
//   T_31 = (N_32 N_21 - N_31) Xd_1;  T_32 = -N_32 Xd_2,
// every Xd_j taken as Xd_j diag(col_scale_j). Every product takes float16
// operands and accumulates in float32. nb and xb are the exchange tiles; out
// receives T.
template <typename OutT>
__device__ __forceinline__ void gdn_chunk_inverse(const float* l,
                                                  const float* col_scale,
                                                  half_t* nb, half_t* xb,
                                                  OutT* out, int tx,
                                                  int barrier_id) {
  int warp = (tx % kGdnGroupThreads) >> 5;
  int lane = tx & 31;
  // Warp w's diagonal block, registers 8 w to 8 w + 7, selected with fixed
  // indices so every array stays in registers.
  float d[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = 0.0f;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
#pragma unroll
    for (int i = 0; i < 8; ++i) d[i] = (w == warp) ? l[8 * w + i] : d[i];
  }
  float x[8];
  uint32_t ax[4];
  gdn_diag_inverse(d, x, lane);
  gdn_pack16(x, ax);

  int c = (lane & 3) << 1;
  uint32_t axs[4];
  {
    float xs[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      xs[i] = x[i] * col_scale[16 * warp + c + (i & 1) + (i >> 2) * 8];
    gdn_pack16(xs, axs);
  }

  // N_wj for j < w: Lo_wj's eight-column halves, each two 8 x 8 tiles
  // transposed into the B operand.
  float n[24];
  uint32_t an[12];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
#pragma unroll
    for (int i = 0; i < 8; ++i) n[8 * j + i] = 0.0f;
    if (j < warp) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int b = 2 * j + h;
        uint32_t bu = gdn_transpose8(gdn_pack_half(l[4 * b], l[4 * b + 1]));
        uint32_t bl = gdn_transpose8(gdn_pack_half(l[4 * b + 2], l[4 * b + 3]));
        gdn_mma16816(n + 8 * j + 4 * h, ax, bu, bl);
      }
    }
    gdn_pack16(n + 8 * j, an + 4 * j);
  }
  if (warp < 3) {
    gdn_store_a16(xb, warp, warp, axs, lane);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      if (j < warp) gdn_store_a16(nb, warp, j, an + 4 * j, lane);
    }
  }
  gdn_group_sync(barrier_id);

  // Row block w of T, block j in t[8 j .. 8 j + 7]; the sign flip of a packed
  // pair is an XOR.
  constexpr uint32_t kNegHalf2 = 0x80008000u;
  float t[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) t[i] = 0.0f;
  if (warp == 1) {
    uint32_t az[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) az[i] = an[i] ^ kNegHalf2;
    gdn_mma_tile(t, az, xb, 0, 0, lane);
  } else if (warp == 2) {
    float z[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) z[i] = -n[i];
    gdn_mma_tile(z, an + 4, nb, 1, 0, lane);
    uint32_t az[4];
    gdn_pack16(z, az);
    gdn_mma_tile(t, az, xb, 0, 0, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) az[i] = an[4 + i] ^ kNegHalf2;
    gdn_mma_tile(t + 8, az, xb, 1, 1, lane);
  } else if (warp == 3) {
    // m = N_32 N_21; z = N_31 - m.
    float m[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = 0.0f;
    gdn_mma_tile(m, an + 8, nb, 2, 1, lane);
    float z[8];
    uint32_t az[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) z[i] = n[8 + i] - m[i];
    gdn_pack16(z, az);
    float z0[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) z0[i] = -n[i];
    gdn_mma_tile(z0, az, nb, 1, 0, lane);
    gdn_mma_tile(z0, an + 8, nb, 2, 0, lane);
    uint32_t az0[4];
    gdn_pack16(z0, az0);
    gdn_mma_tile(t, az0, xb, 0, 0, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) az[i] ^= kNegHalf2;
    gdn_mma_tile(t + 8, az, xb, 1, 1, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) az[i] = an[8 + i] ^ kNegHalf2;
    gdn_mma_tile(t + 16, az, xb, 2, 2, lane);
  }
  // Every warp has read the exchange tiles before any warp writes the next
  // chunk's.
  gdn_group_sync(barrier_id);
  // T_ww = Xd_w, placed with fixed indices so t stays in registers.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = gdn_unpack_half(axs[i]);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      if (w == warp) {
        t[8 * w + 2 * i] = f.x;
        t[8 * w + 2 * i + 1] = f.y;
      }
    }
  }
  gdn_store64(t, out, tx);
}

// T and P for one chunk, by one warp group. gate holds the chunk's cumulative
// gates in log2 units, beta its step sizes; rows past a sequence's end carry a
// zero step size.
//
// P forms transposed, as K Q^T, so each place of a warp's rows takes one gate
// factor: Lg's below the diagonal, P^T's on and above it. Every exponent is
// clamped at zero, so a gate that rises inside a chunk cannot overflow.
template <tl::DataType DT, typename InT>
__device__ __forceinline__ void gdn_chunk_tp(const InT* q, const InT* k,
                                             const float* gate,
                                             const float* beta, float scale,
                                             half_t* nb, half_t* xb, InT* t_out,
                                             InT* p_out, int tx,
                                             int barrier_id) {
  static_assert(DT == tl::DataType::kBFloat16 || DT == tl::DataType::kFloat16,
                "gdn_chunk_tp takes bfloat16 or float16 tiles");
  tx = gdn_opaque(tx);
  float kk[32], pt[32];
  {
    tl::GmmaDescriptor desc_q, desc_k;
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_q, q);
    tl::initialize_wgmma_descriptor<1, 1, 64>(desc_k, k);
    tl::warpgroup_fence_operand(kk, 32);
    tl::warpgroup_fence_operand(pt, 32);
    tl::warpgroup_arrive();
#pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
      int off = (((ki >> 2) * kGdnAtomBytes) + ((ki & 3) * 32)) >> 4;
      tl::wgmma_ss<DT, DT, tl::DataType::kFloat32, 64, 64, 16, false, false, 1,
                   1>(uint64_t(desc_k + off), uint64_t(desc_k + off),
                      reinterpret_cast<uint32_t*>(kk), ki > 0);
    }
#pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
      int off = (((ki >> 2) * kGdnAtomBytes) + ((ki & 3) * 32)) >> 4;
      tl::wgmma_ss<DT, DT, tl::DataType::kFloat32, 64, 64, 16, false, false, 1,
                   1>(uint64_t(desc_k + off), uint64_t(desc_q + off),
                      reinterpret_cast<uint32_t*>(pt), ki > 0);
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(kk, 32);
    tl::warpgroup_fence_operand(pt, 32);
  }
  int warp = (tx % kGdnGroupThreads) >> 5;
  int lane = tx & 31;
  int r0 = warp * 16 + (lane >> 2);
  float g0 = gate[r0];
  float g1 = gate[r0 + 8];
  float b0 = beta[r0];
  float b1 = beta[r0 + 8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float2 gc =
        *reinterpret_cast<const float2*>(gate + 8 * j + ((lane & 3) << 1));
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      int col = 8 * j + ((lane & 3) << 1) + e;
      float gcol = e == 0 ? gc.x : gc.y;
      bool low0 = r0 > col;
      bool low1 = r0 + 8 > col;
      float f0 = exp2f(fminf(low0 ? g0 - gcol : gcol - g0, 0.0f));
      float f1 = exp2f(fminf(low1 ? g1 - gcol : gcol - g1, 0.0f));
      kk[4 * j + e] *= low0 ? b0 * f0 : 0.0f;
      kk[4 * j + 2 + e] *= low1 ? b1 * f1 : 0.0f;
      pt[4 * j + e] *= low0 ? 0.0f : scale * f0;
      pt[4 * j + 2 + e] *= low1 ? 0.0f : scale * f1;
    }
  }
  // P^T's 8 x 8 tile (rows 16 w + 8 m, columns 8 j) lands transposed at P's
  // rows 8 j, columns 16 w + 8 m.
  {
    int m = (lane >> 3) & 1;
    char* base = reinterpret_cast<char*>(p_out);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t r0p = gdn_pack_out(pt[4 * j], pt[4 * j + 1], p_out);
      uint32_t r1p = gdn_pack_out(pt[4 * j + 2], pt[4 * j + 3], p_out);
      uint32_t addr = smem_ptr_to_uint(
          base + gdn_swizzle(8 * j + (lane & 7), 16 * warp + 8 * m));
      asm volatile(
          "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::
              "r"(addr),
          "r"(r0p), "r"(r1p));
    }
  }
  gdn_chunk_inverse<InT>(kk, beta, nb, xb, t_out, tx, barrier_id);
  // T and P are read by wgmma, through the async proxy.
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void gdn_chunk_tp_bf16(bfloat16_t* q, bfloat16_t* k,
                                                  float* gate, float* beta,
                                                  float scale, half_t* nb,
                                                  half_t* xb, bfloat16_t* t_out,
                                                  bfloat16_t* p_out, int tx,
                                                  int barrier_id) {
  gdn_chunk_tp<tl::DataType::kBFloat16>(q, k, gate, beta, scale, nb, xb, t_out,
                                        p_out, tx, barrier_id);
}

__device__ __forceinline__ void gdn_chunk_tp_fp16(
    half_t* q, half_t* k, float* gate, float* beta, float scale, half_t* nb,
    half_t* xb, half_t* t_out, half_t* p_out, int tx, int barrier_id) {
  gdn_chunk_tp<tl::DataType::kFloat16>(q, k, gate, beta, scale, nb, xb, t_out,
                                       p_out, tx, barrier_id);
}

}  // namespace tileops
