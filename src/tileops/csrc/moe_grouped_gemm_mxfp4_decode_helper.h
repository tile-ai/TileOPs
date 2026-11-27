#pragma once

// Run lookup, decode, contraction, promotion and store of the MXFP4 grouped
// GEMM's decode kernel: one 64-row weight tile of one expert times Tiles
// tiles of RowTile rows of ``a``, the weights decoded once a K step to e4m3
// register fragments and contracted as moe_grouped_gemm_mxfp4_helper.h's
// register-fragment path does. RowTile is the fp8 WGMMA's least N: of each
// tile, a math thread holds the accumulators of tokens 2 (lane % 4) and the
// next.

#include "moe_grouped_gemm_mxfp4_helper.h"

namespace tileops {

// Whether an expert's run of rows [lo, hi) holds row ``row``, from ``ends``
// (each of Groups experts' end row) scanned by the block's Threads threads,
// every load issued before any is used; every thread of the block calls it.
// ``run`` gets (expert, lo, hi) through the three ints of ``shared``.
template <int Threads, int Groups>
__device__ __forceinline__ bool mxfp4_decode_find_run(int const* ends, int row,
                                                      int* shared, int* run) {
  constexpr int kPer = (Groups + Threads - 1) / Threads;
  int lo[kPer];
  int hi[kPer];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    int const g = static_cast<int>(threadIdx.x) + i * Threads;
    lo[i] = g > 0 && g < Groups ? ends[g - 1] : 0;
    hi[i] = g < Groups ? ends[g] : 0;
  }
  bool found = false;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    int const g = static_cast<int>(threadIdx.x) + i * Threads;
    if (g < Groups && lo[i] <= row && row < hi[i]) {
      shared[0] = g;
      shared[1] = lo[i];
      shared[2] = hi[i];
      found = true;
    }
  }
  bool const any = __syncthreads_or(found) != 0;
  run[0] = shared[0];
  run[1] = shared[1];
  run[2] = shared[2];
  return any;
}

// The token scales of Rows rows of ``a`` from ``row0`` as floats,
// ``tokens[Rows kk + t]`` for row row0 + t at K step kk, a row past ``rows``
// reading the last; thread ``thread`` of ``Threads`` fills its share.
template <int Steps, int Threads, int Rows>
__device__ __forceinline__ void mxfp4_decode_tokens(uint8_t const* a_scale,
                                                    int row0, int rows,
                                                    float* tokens, int thread) {
#pragma unroll
  for (int i = thread; i < Rows * Steps; i += Threads) {
    int const row = min(row0 + i % Rows, rows - 1);
    tokens[i] =
        mxfp4_e8m0_value(a_scale[static_cast<int64_t>(row) * Steps + i / Rows]);
  }
}

// mxfp4_weight_step under kMxfp4Fold, its codes word ``step`` of each row of
// ``codes`` (rows of CodeSteps words), each row scale 2 ** (top - 127) formed
// by a shift: the fragments and scales of a step whose codes lie in
// [kMxfp4CodeLow, kMxfp4CodeHigh]. ``fits`` is cleared unless both of the
// thread's rows' four codes lie there and span at most kMxfp4FoldSpan.
template <int CodeSteps>
__device__ __forceinline__ void mxfp4_decode_fold_step(
    void const* packed_stage, void const* codes, int step, uint32_t* frags,
    float* scales, int* fits) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const g = ((static_cast<int>(threadIdx.x) >> 5) & 3) * 16 + lane / 4;
  int const offset = 2 * (lane & 3);
  int const swizzle = (g >> 1) & 3;
  uint8_t const* rows[2] = {
      static_cast<uint8_t const*>(packed_stage) + g * 64 + offset,
      static_cast<uint8_t const*>(packed_stage) + (g + 8) * 64 + offset};
  uint2 folds[2][4];
  bool fold = true;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    uint32_t const word =
        static_cast<uint32_t const*>(codes)[(g + 8 * r) * CodeSteps + step];
    uint32_t c[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c[i] = __byte_perm(word, 0, 0x4440u + i);
    }
    uint32_t const top = max(max(c[0], c[1]), max(c[2], c[3]));
    uint32_t const low = min(min(c[0], c[1]), min(c[2], c[3]));
    fold = fold && top - low <= kMxfp4FoldSpan && mxfp4_code_fast(top) &&
           mxfp4_code_fast(low);
    scales[r] = __uint_as_float(top << 23);
    uint32_t const lo = kMxfp4MagnitudesLo - top * kMxfp4BinadeLo;
    uint32_t const hi = kMxfp4MagnitudesHi - top * kMxfp4BinadeHi;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      folds[r][i] =
          make_uint2(lo + c[i] * kMxfp4BinadeLo, hi + c[i] * kMxfp4BinadeHi);
    }
  }
  *fits = *fits & static_cast<int>(fold);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      uint8_t const* chunk = rows[r] + 16 * (i ^ swizzle);
      uint32_t const lo = *reinterpret_cast<uint16_t const*>(chunk);
      uint32_t const hi = *reinterpret_cast<uint16_t const*>(chunk + 8);
      frags[4 * i + r] = mxfp4_e2m1x4_e4m3(lo, folds[r][i]);
      frags[4 * i + 2 + r] = mxfp4_e2m1x4_e4m3(hi, folds[r][i]);
    }
  }
}

// mxfp4_wgmma_rs<N, First, Count, 1> on the N token rows of the stage at
// ``stage``, its descriptor formed from the stage's shared address.
template <int N, int First, int Count>
__device__ __forceinline__ void mxfp4_decode_wgmma(float* accumulator,
                                                   uint32_t* frags,
                                                   void const* stage) {
  uint32_t const lo =
      ((static_cast<uint32_t>(__cvta_generic_to_shared(stage)) >> 4) &
       0x3fffu) |
      0x10000u;
  mxfp4_wgmma_rs<N, First, Count, 1>(accumulator, frags, lo);
}

// final += step * row scale * token scale over the warp-group's 64 x RowTile
// accumulators of the first ``tiles`` of Tiles token tiles: the thread's
// weight rows g and g + 8, and its tokens 2 (lane % 4) and the next of each
// tile from ``tokens``, the K step's Tiles * RowTile token scales. Folded,
// ``step`` is the chain of the step's four sub-blocks and ``scales`` the
// rows' two; split, it is sub-block ``sub`` alone and ``scales`` the rows'
// four each, applied by mxfp4_scaled_exact.
template <int RowTile, int Tiles, bool Fold>
__device__ __forceinline__ void mxfp4_decode_promote(float const* step,
                                                     float* final,
                                                     float const* scales,
                                                     float const* tokens,
                                                     int tiles, int sub) {
  static_assert(RowTile == 8, "a thread holds 4 accumulators of a tile");
  int const pair = (static_cast<int>(threadIdx.x) & 3) * 2;
#pragma unroll
  for (int t = 0; t < Tiles; ++t) {
    if (t < tiles) {
      float const token[2] = {tokens[t * RowTile + pair],
                              tokens[t * RowTile + pair + 1]};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        if constexpr (Fold) {
          final[4 * t + e] =
              final[4 * t + e] + step[4 * t + e] * scales[e / 2] * token[e % 2];
        } else {
          final[4 * t + e] =
              final[4 * t + e] + mxfp4_scaled_exact(step[4 * t + e],
                                                    scales[e / 2 * 4 + sub],
                                                    token[e % 2]);
        }
      }
    }
  }
}

// ``value`` into ``flag`` (shared), then an arrival on ``barrier``, which
// releases the store to a thread that waits on it.
__device__ __forceinline__ void mxfp4_decode_publish(int* flag, int value,
                                                     void* barrier) {
  *reinterpret_cast<int volatile*>(flag) = value;
  uint32_t const bar = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(bar)
               : "memory");
}

// The value mxfp4_decode_publish stored in ``flag``, read after a wait on
// its barrier.
__device__ __forceinline__ int mxfp4_decode_read(int const* flag) {
  return *reinterpret_cast<int const volatile*>(flag);
}

// The warp-group's 64 x RowTile accumulators of Tiles token tiles in bfloat16
// into the first ``valid`` rows of the rows at ``tile`` (rows of ``n``
// columns): token t's columns g and g + 8.
template <int RowTile, int Tiles>
__device__ __forceinline__ void mxfp4_decode_store(float const* final,
                                                   void* tile, int n,
                                                   int valid) {
  static_assert(RowTile == 8, "a thread holds 4 accumulators of a tile");
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const col = ((static_cast<int>(threadIdx.x) >> 5) & 3) * 16 + lane / 4;
  int const pair = (lane & 3) * 2;
  __nv_bfloat16* c = static_cast<__nv_bfloat16*>(tile);
#pragma unroll
  for (int t = 0; t < Tiles; ++t) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int const row = t * RowTile + pair + (e & 1);
      if (row < valid) {
        c[static_cast<int64_t>(row) * n + col + (e >= 2 ? 8 : 0)] =
            __float2bfloat16_rn(final[4 * t + e]);
      }
    }
  }
}

}  // namespace tileops
