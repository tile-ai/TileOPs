#pragma once

// Device functions GemmW4A8DecodeKernel (tileops/kernels/gemm/w4a8_decode.py)
// calls through T.call_extern. An array parameter's extent is the length of the
// local array the kernel passes, so a mismatch fails to compile.
//
// Weight encoding (W4A8RepackFwdInterface order): a 512-byte unit holds 16 rows
// x 64 K; lane l = 4 g + t owns bytes 16 l .. 16 l + 15, word j = 2 p + h holds
// row g + 8 h of K32 step p, nibble e of a word is k = 32 p + 16 (e / 4) + 4 t
// + e % 4, in the nibble code repack_w4a8_weight defines; bit 3 is the sign.
//
// For a row with fp8 group scale s, Tm[i] = e4m3_rne(|s| * (i + 1)), i = 0..7.
// dequant_step's plus fragment holds Tm[|q| - 1] where q * s > 0 and 0
// elsewhere, its minus fragment Tm[|q| - 1] where q * s < 0; with the minus
// WGMMA issued at scale-a = -1, the weight entering the product is
// e4m3_rne(q * s).

#include <cuda_bf16.h>
#include <tl_templates/cuda/barrier.h>

#include <cstdint>

namespace tileops {
namespace w4a8 {

// Warp and warpgroup sizes: ISA facts.
constexpr int kWarpLanes = 32;
constexpr int kWarpgroupThreads = 128;

__device__ __forceinline__ uint32_t shared_address(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// Per item: this lane builds the tables of group t % NG for its rows srow0 and
// srow0 + 8 from the stage's raw scale bytes ([row][group], NG per row) at
// scales + stage_bytes. L = {a0, a1, b0, b1, scale bytes}: a0, a1 = Tm[0..3],
// Tm[4..7] of row srow0, b0, b1 those of row srow0 + 8. Every product |s| * m
// with m <= 8 is exact in fp16, so the e4m3 conversion is the only rounding.
// Returns 1 when any lane of the warp holds a negative scale.
template <int NG>
__device__ __forceinline__ int item_tables(uint32_t (&L)[5], const void* scales,
                                           int stage_bytes, int srow0) {
  static_assert(NG == 1 || NG == 2 || NG == 4,
                "one group per quad lane: NG is 1, 2 or 4");
  const uint8_t* sc = static_cast<const uint8_t*>(scales) + stage_bytes;
  const int g = (threadIdx.x % 4) & (NG - 1);
  const uint32_t sb =
      uint32_t(sc[srow0 * NG + g]) | (uint32_t(sc[(srow0 + 8) * NG + g]) << 8);
  uint32_t s;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
      : "=r"(s)
      : "h"(static_cast<uint16_t>(sb & 0x7F7Fu)));
  const uint32_t m[8] = {0x3C003C00u, 0x40004000u, 0x42004200u,
                         0x44004400u, 0x45004500u, 0x46004600u,
                         0x47004700u, 0x48004800u};  // fp16 1..8
  uint32_t e[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t h;
    uint16_t v;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(h) : "r"(s), "r"(m[i]));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(v) : "r"(h));
    e[i] = v;
  }
  const uint32_t q01 = prmt(e[0], e[1], 0x5410), q23 = prmt(e[2], e[3], 0x5410);
  const uint32_t q45 = prmt(e[4], e[5], 0x5410), q67 = prmt(e[6], e[7], 0x5410);
  L[0] = prmt(q01, q23, 0x6420);
  L[2] = prmt(q01, q23, 0x7531);
  L[1] = prmt(q45, q67, 0x6420);
  L[3] = prmt(q45, q67, 0x7531);
  L[4] = sb;
  return __any_sync(0xffffffffu, (sb & 0x8080u) != 0) ? 1 : 0;
}

// Tables of 128-K group gi from the quad lane that built them. G = {a0, a1, b0,
// b1, a1 and b1 with entry 7 cleared, selector flips of a negative scale for
// rows srow0 and srow0 + 8}.
template <bool FLIP>
__device__ __forceinline__ void group_tables(uint32_t (&G)[8],
                                             const uint32_t (&L)[5], int gi) {
  const int src = ((threadIdx.x % kWarpLanes) & ~3) | gi;
  G[0] = __shfl_sync(0xffffffffu, L[0], src);
  G[1] = __shfl_sync(0xffffffffu, L[1], src);
  G[2] = __shfl_sync(0xffffffffu, L[2], src);
  G[3] = __shfl_sync(0xffffffffu, L[3], src);
  G[4] = G[1] & 0x00FFFFFFu;
  G[5] = G[3] & 0x00FFFFFFu;
  if constexpr (FLIP) {
    const uint32_t s = __shfl_sync(0xffffffffu, L[4], src);
    G[6] = (s & 0x80u) ? 0x88888888u : 0u;
    G[7] = (s & 0x8000u) ? 0x88888888u : 0u;
  } else {
    G[6] = 0u;
    G[7] = 0u;
  }
}

// This lane's 16 bytes of K slice s2 of unit row `atom` (16-row group). Shared
// stage layout: [row block u][slice][units_per_slice][UNIT_BYTES]; a stage
// holds SLICES slices of UNITS (units per row block) units per row block, and a
// compact last block has units_per_slice < UNITS. Issued as written, so a load
// placed one slice ahead stays ahead.
template <int UW, int SLICES, int UNITS, int UNIT_BYTES>
__device__ __forceinline__ void load_slice(uint32_t (&v)[4], const void* ring,
                                           int stage, int atom, int s2,
                                           int units_per_slice) {
  static_assert(UNIT_BYTES == kWarpLanes * sizeof(v),
                "a unit is one 16-byte vector load per lane");
  const uint8_t* p =
      static_cast<const uint8_t*>(ring) +
      ((long)stage * UW * SLICES * UNITS + (atom / UNITS) * SLICES * UNITS +
       s2 * units_per_slice + (atom % UNITS)) *
          UNIT_BYTES +
      (threadIdx.x % kWarpLanes) * sizeof(v);
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
               : "r"(shared_address(p)));
}

// Plus (f[0..3]) and minus (f[4..7]) A fragments of K32 step pp (0 or 1) of a
// loaded slice. A selector nibble with bit 3 set makes PRMT replicate a table
// byte's sign bit, 0 for every entry, so the other sign's nibbles give 0x00.
template <bool FLIP>
__device__ __forceinline__ void dequant_step(uint32_t* f,
                                             const uint32_t (&v)[4],
                                             const uint32_t (&G)[8], int pp) {
  const uint32_t w0 = pp ? v[2] : v[0], w1 = pp ? v[3] : v[1];
  const uint32_t a0 = G[0], a1 = G[1], b0 = G[2], b1 = G[3], pa1 = G[4],
                 pb1 = G[5];
  if constexpr (!FLIP) {
    const uint32_t x0 = w0 ^ 0x88888888u, x1 = w1 ^ 0x88888888u;
    f[0] = prmt(a0, pa1, w0);
    f[2] = prmt(a0, pa1, w0 >> 16);
    f[4] = prmt(a0, a1, x0);
    f[6] = prmt(a0, a1, x0 >> 16);
    f[1] = prmt(b0, pb1, w1);
    f[3] = prmt(b0, pb1, w1 >> 16);
    f[5] = prmt(b0, b1, x1);
    f[7] = prmt(b0, b1, x1 >> 16);
  } else {
    const uint32_t fa = G[6], fb = G[7];
    const uint32_t sp0 = w0 ^ fa, sm0 = sp0 ^ 0x88888888u;
    const uint32_t sp1 = w1 ^ fb, sm1 = sp1 ^ 0x88888888u;
    const uint32_t ta_p = fa ? a1 : pa1, ta_m = fa ? pa1 : a1;
    const uint32_t tb_p = fb ? b1 : pb1, tb_m = fb ? pb1 : b1;
    f[0] = prmt(a0, ta_p, sp0);
    f[2] = prmt(a0, ta_p, sp0 >> 16);
    f[4] = prmt(a0, ta_m, sm0);
    f[6] = prmt(a0, ta_m, sm0 >> 16);
    f[1] = prmt(b0, tb_p, sp1);
    f[3] = prmt(b0, tb_p, sp1 >> 16);
    f[5] = prmt(b0, tb_m, sm1);
    f[7] = prmt(b0, tb_m, sm1 >> 16);
  }
}

// Warpgroup index broadcast from lane 0: provably warp-uniform, so WGMMA
// descriptor arithmetic that depends on it stays in uniform registers.
__device__ __forceinline__ int warpgroup_uniform() {
  return __shfl_sync(0xffffffffu,
                     static_cast<int>(threadIdx.x) / kWarpgroupThreads, 0);
}

// K-major, 128-byte-swizzled B descriptor (LBO 16 B, SBO 1024 B) of a shared
// buffer.
template <typename Desc>
__device__ __forceinline__ void swizzled_descriptor(Desc& d, const void* smem) {
  const uint32_t a = __shfl_sync(0xffffffffu, shared_address(smem), 0);
  d.desc_ = (uint64_t)((a >> 4) & 0x3FFF) | ((uint64_t)1 << 16) |
            ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

// Slices [s0, s0 + nsl) of a compact last row block of `units` units
// (units * UNIT_BYTES * nsl contiguous bytes) into the stage at shared address
// ring + stage_bytes, completing on `bar`'s transaction count.
template <int UNIT_BYTES>
__device__ __forceinline__ void load_compact_block(uint32_t ring,
                                                   long stage_bytes,
                                                   const void* block, int s0,
                                                   int nsl, int units,
                                                   Barrier* bar) {
  const uint32_t b = shared_address(bar);
  const uint32_t bytes = (uint32_t)(nsl * units * UNIT_BYTES);
  asm volatile(
      "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(b),
      "r"(bytes));
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];" ::"r"(ring + (uint32_t)stage_bytes),
      "l"(static_cast<const uint8_t*>(block) + (long)s0 * units * UNIT_BYTES),
      "r"(bytes), "r"(b)
      : "memory");
}

// Scale bytes of item `item` (row tile item / nchunk, K chunk item % nchunk)
// for this lane's ROWS / 32 rows tile * ROWS + h * 32 + lane, one word per row
// whose byte j is group NG * (item % nchunk) + j, 0 past the last group. Rows
// past nrows read row nrows - 1, whose outputs are never stored; the loads stay
// unconditional so they issue early.
template <int ROWS, int NG>
__device__ __forceinline__ void load_item_scales(
    uint32_t (&r)[ROWS / kWarpLanes], const void* gs_, int groups, int item,
    int nchunk, int nrows) {
  static_assert(NG <= sizeof(r[0]), "a row's NG scale bytes fill one word");
  const uint8_t* gs = static_cast<const uint8_t*>(gs_);
  const int lane = threadIdx.x % kWarpLanes;
  const int tile = item / nchunk, g0 = (item - tile * nchunk) * NG;
  const int row0 = tile * ROWS;
  const int ng = min(NG, groups - g0);
#pragma unroll
  for (int h = 0; h < ROWS / kWarpLanes; ++h) {
    const int row = min(row0 + h * kWarpLanes + lane, nrows - 1);
    const uint8_t* q = gs + (long)row * groups + g0;
    uint32_t v = 0;
#pragma unroll
    for (int j = 0; j < NG; ++j)
      if (j < ng) v |= uint32_t(__ldg(q + j)) << (8 * j);
    r[h] = v;
  }
}

// Writes load_item_scales' words as the stage's [row][NG] scale bytes.
template <int ROWS, int NG>
__device__ __forceinline__ void store_item_scales(
    void* sc_, const uint32_t (&r)[ROWS / kWarpLanes]) {
  static_assert(NG == 1 || NG == 2 || NG == 4, "NG is 1, 2 or 4");
  uint8_t* sc = static_cast<uint8_t*>(sc_);
  const int lane = threadIdx.x % kWarpLanes;
#pragma unroll
  for (int h = 0; h < ROWS / kWarpLanes; ++h) {
    const int row = h * kWarpLanes + lane;
    if constexpr (NG == 4)
      reinterpret_cast<uint32_t*>(sc)[row] = r[h];
    else if constexpr (NG == 2)
      reinterpret_cast<uint16_t*>(sc)[row] = (uint16_t)r[h];
    else
      sc[row] = (uint8_t)r[h];
  }
}

// One element of `base`, loaded through the read-only path and issued where it
// is written.
__device__ __forceinline__ float load_early(const float* base, int idx) {
  float v;
  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(base + idx));
  return v;
}

// WGMMA accumulator layout: element c of lane 4 g + t is row g + 8 ((c >> 1)
// & 1) of the warp's 16, token 8 (c >> 2) + 2 t + (c & 1). Stages the
// warpgroup's [token][64 rows] bf16 tile (row stride STRIDE) between two named
// barriers `bar_id` of its threads, then writes 16-byte row pieces of tokens
// < M and rows < N.
template <int NT, int STRIDE>
__device__ __forceinline__ void store_staged(const float (&v)[NT / 2],
                                             void* out_, int M, int N,
                                             long row0, int rl0, float ch0,
                                             float ch1, const float* as_s,
                                             void* stage_, int bar_id) {
  constexpr int kWgmmaRows = 64;  // M extent of the FP8 WGMMA instruction.
  constexpr int kPieceRows = sizeof(uint4) / sizeof(__nv_bfloat16);
  constexpr int kPieces = kWgmmaRows / kPieceRows;
  static_assert(STRIDE >= kWgmmaRows && STRIDE % kPieceRows == 0,
                "a staged token row holds the tile's rows in 16-byte pieces");
  __nv_bfloat16* out = static_cast<__nv_bfloat16*>(out_);
  __nv_bfloat16* stage = static_cast<__nv_bfloat16*>(stage_);
  const int tw = threadIdx.x % kWarpgroupThreads, t = threadIdx.x % 4;
  asm volatile("bar.sync %0, %1;" ::"r"(bar_id), "n"(kWarpgroupThreads));
#pragma unroll
  for (int c = 0; c < NT / 2; ++c) {
    const int tok = (c >> 2) * 8 + 2 * t + (c & 1);
    stage[tok * STRIDE + rl0 + 8 * ((c >> 1) & 1)] =
        __float2bfloat16_rn(v[c] * ((c & 2) ? ch1 : ch0) * as_s[tok]);
  }
  asm volatile("bar.sync %0, %1;" ::"r"(bar_id), "n"(kWarpgroupThreads));
  for (int j = tw; j < NT * kPieces; j += kWarpgroupThreads) {
    const int tok = j / kPieces, piece = j % kPieces;
    if (tok < M && row0 + piece * kPieceRows < N)
      *reinterpret_cast<uint4*>(out + (long)tok * N + row0 +
                                piece * kPieceRows) =
          *reinterpret_cast<const uint4*>(stage + tok * STRIDE +
                                          piece * kPieceRows);
  }
}

// Thread-contiguous fp32 partial sums of one warpgroup: slot[tw * NT / 2 + q],
// shared or global. The caller's CTA barriers order the slot's writes and
// reads.
template <int NT>
__device__ __forceinline__ void park(float* slot, const float (&v)[NT / 2]) {
  float4* d = reinterpret_cast<float4*>(
      slot + (threadIdx.x % kWarpgroupThreads) * (NT / 2));
#pragma unroll
  for (int q = 0; q < NT / 8; ++q)
    d[q] = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
}
// Adds a slot park wrote into sum.
template <int NT>
__device__ __forceinline__ void add_parked(float (&sum)[NT / 2],
                                           const float* slot) {
  const float4* s = reinterpret_cast<const float4*>(
      slot + (threadIdx.x % kWarpgroupThreads) * (NT / 2));
#pragma unroll
  for (int q = 0; q < NT / 8; ++q) {
    const float4 x = s[q];
    sum[4 * q] += x.x;
    sum[4 * q + 1] += x.y;
    sum[4 * q + 2] += x.z;
    sum[4 * q + 3] += x.w;
  }
}

// Hands one warpgroup's partial sums to the CTA that owns the tile: writes them
// to the global slot, then sets `flag` to 1 with release semantics. A CTA
// publishes at most once per launch, and only the owner reads its flag.
template <int NT>
__device__ __forceinline__ void publish(float* slot, int* flag,
                                        const float (&v)[NT / 2], int bar_id) {
  const int tw = threadIdx.x % kWarpgroupThreads;
  float4* d = reinterpret_cast<float4*>(slot + tw * (NT / 2));
#pragma unroll
  for (int q = 0; q < NT / 8; ++q)
    __stcg(d + q,
           make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
  __threadfence();
  asm volatile("bar.sync %0, %1;" ::"r"(bar_id), "n"(kWarpgroupThreads)
               : "memory");
  if (tw == 0)
    asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(flag), "r"(1)
                 : "memory");
}

// Adds every part published for the tile whose first item is item0: the parts
// of CTAs [first CTA holding item0, cid). Waits for each flag (acquire), then
// clears it to 0, so the flags are zero again when the launch ends and a
// captured graph replays without resetting them.
template <int NT>
__device__ __forceinline__ void receive_all(float (&sum)[NT / 2],
                                            const float* xs, int* xf, int item0,
                                            int items, int ctas, int cid,
                                            int uw, int sub, int bar_id) {
  const int tw = threadIdx.x % kWarpgroupThreads;
  int c = (int)((long)item0 * ctas / items);
  while ((long)(c + 1) * items / ctas <= item0) ++c;
  while ((long)c * items / ctas > item0) --c;
  for (; c < cid; ++c) {
    int* flag = xf + c * uw + sub;
    if (tw == 0) {
      int seen;
      do {
        asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
                     : "=r"(seen)
                     : "l"(flag)
                     : "memory");
      } while (seen != 1);
      *flag = 0;
    }
    asm volatile("bar.sync %0, %1;" ::"r"(bar_id), "n"(kWarpgroupThreads)
                 : "memory");
    const float4* s = reinterpret_cast<const float4*>(
        xs + ((long)c * uw + sub) * kWarpgroupThreads * (NT / 2) +
        tw * (NT / 2));
#pragma unroll
    for (int q = 0; q < NT / 8; ++q) {
      const float4 x = __ldcg(s + q);
      sum[4 * q] += x.x;
      sum[4 * q + 1] += x.y;
      sum[4 * q + 2] += x.z;
      sum[4 * q + 3] += x.w;
    }
  }
}

}  // namespace w4a8
}  // namespace tileops
