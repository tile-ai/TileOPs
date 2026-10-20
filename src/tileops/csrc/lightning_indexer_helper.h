#pragma once

namespace tileops {

// The epilogue of one 64-key tile of the lightning indexer over its BQ queries,
// for thread t of the consumer warpgroup that holds the tile's m64n(HEADS * BQ)
// WGMMA accumulator.
//
// Accumulator element 4 j + 2 h + b of thread t sits at key row 16 (t / 32) +
// (t % 32) / 4
// + 8 h and query-head column 8 j + 2 (t % 4) + b. ``wt`` holds that column's
// head weight at 2 j + b. Each (row, query) logit is the head sum of
// relu(score) * weight, one FMNMX and one FFMA an element. The four lanes that
// share a row then reduce-scatter the 2 BQ sums, so lane t % 4 ends with the
// (query q, row half h) of 2 q + h congruent to it, and stores them times the
// key's scale. Outside its row's window a logit is -inf when CLEAN;
// ``interior`` says the whole tile is inside every window and inside the
// tensor.
template <int HEADS, int BQ, bool CLEAN>
__device__ __forceinline__ void indexer_tile_epilogue(
    const float* acc, const float* wt, const float* scale, float* logits,
    int s0, int k0, int seq_len, int seq_len_kv, const int* win, bool interior,
    int t) {
  constexpr int N = 2 * BQ;  // (query, row half) sums a thread holds
  static_assert(N % 4 == 0, "four lanes share a row's sums");
  constexpr int JQ = HEADS / 8;  // column blocks of 8 a query
  const int lane4 = t % 4;
  const int row0 = 16 * (t / 32) + (t % 32) / 4;
  // Partial sums first: N independent pairs of FFMA chains. Sum 2 q + h.
  float v[N];
#pragma unroll
  for (int q = 0; q < BQ; ++q) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float a = 0.f, b = 0.f;
#pragma unroll
      for (int jj = 0; jj < JQ; ++jj) {
        const int j = q * JQ + jj;
        a = fmaf(fmaxf(acc[4 * j + 2 * h], 0.f), wt[2 * j], a);
        b = fmaf(fmaxf(acc[4 * j + 2 * h + 1], 0.f), wt[2 * j + 1], b);
      }
      v[2 * q + h] = a + b;
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
#pragma unroll
  for (int m = 0; m < N / 4; ++m) {
    const int i = 4 * m + lane4;
    const int q = i / 2;
    const int row = row0 + 8 * (i % 2);
    const int key = k0 + row;
    const int qrow = s0 + q;
    float out = own[m] * scale[row];
    if (interior) {
      logits[static_cast<long long>(qrow) * seq_len_kv + key] = out;
    } else if (key < seq_len_kv && qrow < seq_len) {
      if (CLEAN && (key < win[q] || key >= win[BQ + q])) {
        out = -__int_as_float(0x7f800000);
      }
      logits[static_cast<long long>(qrow) * seq_len_kv + key] = out;
    }
  }
}

}  // namespace tileops
