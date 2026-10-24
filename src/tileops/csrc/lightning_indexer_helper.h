#pragma once

namespace tileops {

// The epilogue of one 64-key tile of the lightning indexer over its BQ queries,
// for thread t of the consumer warpgroup that holds the tile's m64n(HEADS * BQ)
// WGMMA accumulator.
//
// Accumulator element 4*j + 2*h + b belongs to key row
// 16*(t/32) + (t%32)/4 + 8*h and head column 8*j + 2*(t%4) + b.
// wt[2*j+b] is the unscaled head weight; halving it can flush a normal
// FP32 weight to zero before multiplication by the score and key scale.
// Each four-lane group owns a row. key_scale belongs to the stored row
// 16*(t/32) + (t%32)/4 + 8*(t%2). CLEAN masks logits outside the window.
template <int HEADS, int BQ, bool CLEAN>
__device__ __forceinline__ void indexer_tile_epilogue(
    const float* acc, const float* wt, float key_scale, float* logits,
    long long off, int s0, int k0, int seq_len, int seq_len_kv, const int* win,
    bool interior, int t) {
  constexpr int N = 2 * BQ;  // (query, row half) sums a thread holds
  static_assert(N % 4 == 0, "four lanes share a row's sums");
  constexpr int JQ = HEADS / 8;  // column blocks of 8 a query
  const int lane4 = t % 4;
  const int row0 = 16 * (t / 32) + (t % 32) / 4;
  float v[N];
#pragma unroll
  for (int q = 0; q < BQ; ++q) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      // Two chains a sum below 16 sums a thread, so at least 16 FFMA chains run
      // at once.
      float a = 0.f, b = 0.f;
#pragma unroll
      for (int jj = 0; jj < JQ; ++jj) {
        const int j = q * JQ + jj;
        const float x0 = acc[4 * j + 2 * h], x1 = acc[4 * j + 2 * h + 1];
        a = fmaf(fmaxf(x0, 0.f), wt[2 * j], a);
        if (N < 16) {
          b = fmaf(fmaxf(x1, 0.f), wt[2 * j + 1], b);
        } else {
          a = fmaf(fmaxf(x1, 0.f), wt[2 * j + 1], a);
        }
      }
      v[2 * q + h] = N < 16 ? a + b : a;
    }
  }
  // Reduce-scatter over lane bits 1 then 0: sum i stays with lane i % 4, in
  // slot i / 4.
  float half[N / 2];
#pragma unroll
  for (int m = 0; m < N / 4; ++m) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const float lo = v[4 * m + p], up = v[4 * m + 2 + p];
      const bool keep_up = lane4 & 2;
      half[2 * m + p] = (keep_up ? up : lo) +
                        __shfl_xor_sync(0xffffffffu, keep_up ? lo : up, 2);
    }
  }
  float own[N / 4];
#pragma unroll
  for (int m = 0; m < N / 4; ++m) {
    const bool keep_up = lane4 & 1;
    const float lo = half[2 * m], up = half[2 * m + 1];
    own[m] = (keep_up ? up : lo) +
             __shfl_xor_sync(0xffffffffu, keep_up ? lo : up, 1);
  }
  // Sum 4 m + t % 4 is query 2 m + (t % 4) / 2 at row half t % 2: ``off``
  // locates the first, each next one is two queries further. A row past
  // seq_len gets an address but no store.
  float* out = logits + off + k0;
  const int row = row0 + 8 * (lane4 % 2);
#pragma unroll
  for (int m = 0; m < N / 4; ++m) {
    float v_out = own[m] * key_scale;
    float* p = out + 2LL * m * seq_len_kv;
    if (interior) {
      *p = v_out;
    } else {
      const int q = 2 * m + lane4 / 2;
      const int key = k0 + row;
      if (key < seq_len_kv && s0 + q < seq_len) {
        if (CLEAN && (key < win[q] || key >= win[BQ + q])) {
          v_out = -__int_as_float(0x7f800000);
        }
        *p = v_out;
      }
    }
  }
}

}  // namespace tileops
