#pragma once

// Decode, WGMMA, epilogue, scale-table and stream-K helpers of the W4A8 prefill
// GEMM (tileops/kernels/gemm/w4a8_prefill.py).
//
// The packed weight is in the unified order (W4A8RepackFwdInterface): per
// 64-row block and 128-K group a 4 KB tile; thread t of a consumer warpgroup
// reads its 16 bytes of each 64-K half at byte t * 16; word j = 2p + h of those
// 16 bytes is row r0 + 8h (r0 = 16 * (t / 32) + (t % 32) / 4) at K/32 step p,
// its low 16 bits K 4c..4c+3 and its high 16 bits K 16+4c..16+4c+3 (c = t % 4),
// one nibble per weight in the code repack_w4a8_weight defines.

#include <cuda.h>
#include <cuda_bf16.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

namespace tileops {

__device__ __forceinline__ uint32_t w4a8_prmt(uint32_t a, uint32_t b,
                                              uint32_t c) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decodes one word against its row's 16-byte table t (t byte c = e4m3 value of
// SM2 code c): the low 16 bits into lo, the high 16 bits into hi, four e4m3
// values each.
__device__ __forceinline__ void w4a8_decode_word(uint32_t w, const uint4& t,
                                                 uint32_t& lo, uint32_t& hi) {
  uint32_t idx = w & 0x77777777u;
  uint32_t sel = ((w >> 1) & 0x44444444u) | 0x32103210u;
  uint32_t p0 = w4a8_prmt(t.x, t.y, idx);
  uint32_t n0 = w4a8_prmt(t.z, t.w, idx);
  lo = w4a8_prmt(p0, n0, sel);
  uint32_t idx1 = idx >> 16;
  uint32_t p1 = w4a8_prmt(t.x, t.y, idx1);
  uint32_t n1 = w4a8_prmt(t.z, t.w, idx1);
  hi = w4a8_prmt(p1, n1, sel >> 16);
}

// Decodes this thread's 64 weights of warpgroup wg's 64 x 128 tile into a[16]
// and issues the group's four RS WGMMAs (m64 nN k32) against the 128B-swizzled
// [N tokens][128 K] activation tile at b_base + b_off, accumulating unless
// first. w_block is the tile, half_stride the byte distance of its second 64-K
// half (2048, or W * 512 for a compact last block of W 16-row units); tab the
// stage's 128 tables.
template <int N>
__device__ __forceinline__ void w4a8_decode_mma(float* acc, uint32_t* a,
                                                const uint8_t* w_block,
                                                const uint8_t* tab, int wg,
                                                fp8_e4_t* b_base, int b_off,
                                                int first, int half_stride) {
  int const t = static_cast<int>(threadIdx.x) & 127;
  int const r0 = (t >> 5) * 16 + ((t & 31) >> 2);
  uint4 t0 = *reinterpret_cast<const uint4*>(tab + (wg * 64 + r0) * 16);
  uint4 t1 = *reinterpret_cast<const uint4*>(tab + (wg * 64 + r0 + 8) * 16);
  uint4 h0 = *reinterpret_cast<const uint4*>(w_block + t * 16);
  uint4 h1 = *reinterpret_cast<const uint4*>(w_block + half_stride + t * 16);
  w4a8_decode_word(h0.x, t0, a[0], a[2]);
  w4a8_decode_word(h0.y, t1, a[1], a[3]);
  w4a8_decode_word(h0.z, t0, a[4], a[6]);
  w4a8_decode_word(h0.w, t1, a[5], a[7]);
  w4a8_decode_word(h1.x, t0, a[8], a[10]);
  w4a8_decode_word(h1.y, t1, a[9], a[11]);
  w4a8_decode_word(h1.z, t0, a[12], a[14]);
  w4a8_decode_word(h1.w, t1, a[13], a[15]);
  tl::GmmaDescriptor desc_b;
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, b_base + b_off);
  tl::warpgroup_fence_operand(a, 16);
  tl::warpgroup_fence_operand(acc, N / 2);
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    tl::wgmma_rs<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3,
                 tl::DataType::kFloat32, 64, N, 32, false, false, 1, 1>(
        a + 4 * ki, uint64_t(desc_b + ((ki * 32) >> 4)),
        reinterpret_cast<uint32_t*>(acc), (ki > 0) || !first);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(acc, N / 2);
}

// Keeps a's registers unchanged up to here; called once the WGMMA reading them
// has retired.
__device__ __forceinline__ void w4a8_keep16(uint32_t* a) {
  tl::warpgroup_fence_operand(a, 16);
}

// Writes pass `pass` (PM tokens) of warpgroup wg's accumulator as bf16(acc * cs
// * as) into a 128B-swizzled [PM tokens][128 rows] tile, two [PM][64] halves:
// byte offset of (m, n8 = n / 8) is (n8 / 8) * PM * 128 + m * 128 + ((n8 % 8) ^
// (m % 8)) * 16. act_scale points at the pass's first token; tokens from mlim
// on read the scale of token mlim - 1 (they are never stored).
template <int N, int PM>
__device__ __forceinline__ void w4a8_epilogue(float* acc, bfloat16_t* out_smem,
                                              int wg, int pass,
                                              const float* act_scale, float cs0,
                                              float cs1, int mlim) {
  static_assert(N % PM == 0 && PM % 16 == 0,
                "a pass is whole 16-token stmatrix groups of the tile");
  int const t = static_cast<int>(threadIdx.x) & 127;
  int const lane = t & 31;
  int const w = t >> 5;
  int const c = lane & 3;
  uint32_t const base = smem_ptr_to_uint(out_smem);
  int const q = lane >> 3, rr = lane & 7;
  int const n8 = wg * 8 + w * 2 + (q & 1);
  constexpr int kChunks = PM / 8;
  auto bf16x2_bits = [](float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    __nv_bfloat162_raw const raw = v;
    return uint32_t(raw.x) | (uint32_t(raw.y) << 16);
  };
#pragma unroll
  for (int ii = 0; ii < kChunks; ii += 2) {
    int const i = pass * (PM / 8) + ii;
    int const lm = mlim - 1;
    float const a0 = act_scale[min(ii * 8 + 2 * c, lm)];
    float const a1 = act_scale[min(ii * 8 + 2 * c + 1, lm)];
    float const b0 = act_scale[min(ii * 8 + 8 + 2 * c, lm)];
    float const b1 = act_scale[min(ii * 8 + 8 + 2 * c + 1, lm)];
    uint32_t r0 =
        bf16x2_bits(acc[i * 4 + 0] * cs0 * a0, acc[i * 4 + 1] * cs0 * a1);
    uint32_t r1 =
        bf16x2_bits(acc[i * 4 + 2] * cs1 * a0, acc[i * 4 + 3] * cs1 * a1);
    uint32_t r2 =
        bf16x2_bits(acc[i * 4 + 4] * cs0 * b0, acc[i * 4 + 5] * cs0 * b1);
    uint32_t r3 =
        bf16x2_bits(acc[i * 4 + 6] * cs1 * b0, acc[i * 4 + 7] * cs1 * b1);
    int const m = ii * 8 + (q >> 1) * 8 + rr;
    uint32_t addr =
        base + (n8 >> 3) * (PM * 128) + m * 128 + (((n8 & 7) ^ (m & 7)) << 4);
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, "
        "%4};\n" ::"r"(addr),
        "r"(r0), "r"(r1), "r"(r2), "r"(r3));
  }
}

template <class B>
__device__ __forceinline__ void w4a8_expect_tx(B* bar, int bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(
                   smem_ptr_to_uint(bar)),
               "r"(bytes)
               : "memory");
}

// TMA of the 2D box at (c0, c1) into smem_base + off of every CTA in `mask`,
// completing on each CTA's barrier at the same offset as bar.
template <class B>
__device__ __forceinline__ void w4a8_tma_multicast_2d(const CUtensorMap& desc,
                                                      B* bar, void* smem_base,
                                                      int off, int mask, int c0,
                                                      int c1) {
  void* smem = reinterpret_cast<char*>(smem_base) + off;
  uint64_t d = reinterpret_cast<uint64_t>(&desc);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes."
      "multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(
          smem_ptr_to_uint(smem)),
      "l"(d), "r"(smem_ptr_to_uint(bar)), "r"(c0), "r"(c1), "h"((uint16_t)mask)
      : "memory");
}

// Arrives once on bar in each of the cluster's Cluster CTAs, with .release.cta
// semantics (a .release.cluster arrive adds a cluster-scope fence per call).
template <int Cluster, class B>
__device__ __forceinline__ void w4a8_arrive_cluster(B* bar) {
  uint32_t const a = smem_ptr_to_uint(bar);
#pragma unroll
  for (int cta = 0; cta < Cluster; ++cta) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(r)
                 : "r"(a), "r"(cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(r)
                 : "memory");
  }
}

// rom[s * 16 + c] = e4m3 value of SM2 code c times the e4m3 scale with byte s,
// rounded to nearest even and saturated to 448, for every finite s of either
// sign: (m1..m4) (m5, m6, m7, 0) (-m1..-m4) (-m5..-m8), m_j = e4m3(j * s).
// Every thread of the block takes part.
__device__ __forceinline__ void w4a8_scale_rom_fill(const uint8_t* rom_c) {
  uint8_t* rom = const_cast<uint8_t*>(rom_c);
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    uint32_t sh2;
    asm("{\n .reg .b16 b;\n cvt.u16.u32 b, %1;\n cvt.rn.f16x2.e4m3x2 %0, b;\n}"
        : "=r"(sh2)
        : "r"(static_cast<uint32_t>(i)));
    sh2 = (sh2 & 0xFFFFu) | (sh2 << 16);
    const uint32_t k12 = 0x40003C00u, k34 = 0x44004200u, k56 = 0x46004500u,
                   k78 = 0x48004700u;
    uint32_t p12, p34, p56, p78, A, B;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(p12) : "r"(sh2), "r"(k12));
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(p34) : "r"(sh2), "r"(k34));
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(p56) : "r"(sh2), "r"(k56));
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(p78) : "r"(sh2), "r"(k78));
    asm("{\n .reg .b16 l, h;\n cvt.rn.satfinite.e4m3x2.f16x2 l, %1;\n"
        " cvt.rn.satfinite.e4m3x2.f16x2 h, %2;\n mov.b32 %0, {l, h};\n}"
        : "=r"(A)
        : "r"(p12), "r"(p34));
    asm("{\n .reg .b16 l, h;\n cvt.rn.satfinite.e4m3x2.f16x2 l, %1;\n"
        " cvt.rn.satfinite.e4m3x2.f16x2 h, %2;\n mov.b32 %0, {l, h};\n}"
        : "=r"(B)
        : "r"(p56), "r"(p78));
    uint4 o;
    o.x = A;
    o.y = w4a8_prmt(B, 0u, 0x4210u);
    o.z = A ^ 0x80808080u;
    o.w = B ^ 0x80808080u;
    *reinterpret_cast<uint4*>(rom + i * 16) = o;
  }
}

// A table lane serves rows row0 and row0 + 32 of a 128-row tile; ring[d * 2 +
// j] holds the scale byte of row row0 + 32 * j at the d-th of Depth K steps
// ahead. With Clamp, rows from rlim on read row rlim - 1 (rows past N, whose
// results are never stored).
template <bool Clamp>
__device__ __forceinline__ int w4a8_ring_row(int row, int rlim) {
  return Clamp ? min(row, rlim - 1) : row;
}

// Loads K steps kb .. kb + Depth - 1 into slots 0 .. Depth - 1; steps from ke
// on read step ke - 1.
template <int Depth, bool Clamp>
__device__ __forceinline__ void w4a8_ring_fill(uint32_t* ring,
                                               const fp8_e4_t* gs, int ld,
                                               int row0, int kb, int ke,
                                               int rlim) {
  const uint8_t* g = reinterpret_cast<const uint8_t*>(gs);
#pragma unroll
  for (int d = 0; d < Depth; ++d) {
    int kk = min(kb + d, ke - 1);
#pragma unroll
    for (int j = 0; j < 2; ++j)
      ring[d * 2 + j] =
          __ldg(g + w4a8_ring_row<Clamp>(row0 + 32 * j, rlim) * ld + kk);
  }
}

// Copies the tables of ring slot `slot` from rom into the stage's tables.
__device__ __forceinline__ void w4a8_ring_tables(const uint8_t* tab_stage_c,
                                                 int row0, const uint32_t* ring,
                                                 const uint8_t* rom, int slot) {
  uint8_t* tab_stage = const_cast<uint8_t*>(tab_stage_c);
#pragma unroll
  for (int j = 0; j < 2; ++j)
    *reinterpret_cast<uint4*>(tab_stage + (row0 + 32 * j) * 16) =
        *reinterpret_cast<const uint4*>(rom + ring[slot * 2 + j] * 16);
}

// Shifting ring (even K-step counts): drops slot 0 and loads K step kk_next
// into the last slot.
template <int Depth, bool Clamp>
__device__ __forceinline__ void w4a8_ring_shift(uint32_t* ring,
                                                const fp8_e4_t* gs, int ld,
                                                int row0, int kk_next, int ke,
                                                int rlim) {
  const uint8_t* g = reinterpret_cast<const uint8_t*>(gs);
#pragma unroll
  for (int i = 0; i < (Depth - 1) * 2; ++i) ring[i] = ring[i + 2];
  int kk = min(kk_next, ke - 1);
#pragma unroll
  for (int j = 0; j < 2; ++j)
    ring[(Depth - 1) * 2 + j] =
        __ldg(g + w4a8_ring_row<Clamp>(row0 + 32 * j, rlim) * ld + kk);
}

// Unrolled ring (odd K-step counts): slot `slot` always holds the K step it
// serves; loads K step kk_next into it. `slot` is a literal at every call, so
// the ring stays in registers.
template <bool Clamp>
__device__ __forceinline__ void w4a8_ring_load(uint32_t* ring,
                                               const fp8_e4_t* gs, int ld,
                                               int row0, int kk_next, int ke,
                                               int rlim, int slot) {
  const uint8_t* g = reinterpret_cast<const uint8_t*>(gs);
  int kk = min(kk_next, ke - 1);
#pragma unroll
  for (int j = 0; j < 2; ++j)
    ring[slot * 2 + j] =
        __ldg(g + w4a8_ring_row<Clamp>(row0 + 32 * j, rlim) * ld + kk);
}

// Stores this thread's R accumulators at ws_c + off, the caller's off ending in
// lt * 4 for consumer thread lt of Threads: float4 chunk j at j * Threads * 4
// floats past it, through L2. The fence orders the stores before the caller's
// w4a8_signal.
template <int R, int Threads>
__device__ __forceinline__ void w4a8_store_partial(const float* acc,
                                                   const float* ws_c, int off) {
  float* p = const_cast<float*>(ws_c) + off;
#pragma unroll
  for (int j = 0; j < R / 4; ++j)
    __stcg(reinterpret_cast<float4*>(p + j * Threads * 4),
           make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2],
                       acc[4 * j + 3]));
  __threadfence();
}

// Adds to acc the partial w4a8_store_partial<R, Threads> stored at ws + off.
template <int R, int Threads>
__device__ __forceinline__ void w4a8_add_partial(float* acc, const float* ws,
                                                 int off) {
  const float* p = ws + off;
#pragma unroll
  for (int j = 0; j < R / 4; ++j) {
    float4 v = __ldcg(reinterpret_cast<const float4*>(p + j * Threads * 4));
    acc[4 * j] += v.x;
    acc[4 * j + 1] += v.y;
    acc[4 * j + 2] += v.z;
    acc[4 * j + 3] += v.w;
  }
}

// Counts one contribution to tile idx; the last of nc resets the counter and
// returns 1.
__device__ __forceinline__ int w4a8_signal(const int* cnt_base, int idx,
                                           int nc) {
  int* cnt = const_cast<int*>(cnt_base) + idx;
  __threadfence();
  int old = atomicAdd(cnt, 1);
  if (old == nc - 1) {
    *cnt = 0;
    __threadfence();
    return 1;
  }
  return 0;
}

}  // namespace tileops
