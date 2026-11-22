#pragma once

namespace tileops {

// Half of head weight ``w``, kept to within one subnormal ulp where it is
// subnormal: the kernel compiles with denormal flushing, which would drop the
// half of a normal weight below 2^-125.
__device__ __forceinline__ float indexer_half_weight(float w) {
  float half;
  asm("mul.rn.f32 %0, %1, 0f3F000000;" : "=f"(half) : "f"(w));
  return half;
}

// a * b + c without flushing a subnormal operand or result.
__device__ __forceinline__ float indexer_fma(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

// The epilogue of one 64-key tile of the lightning indexer over its BQ queries,
// for thread t of the consumer warpgroup that holds the tile's m64n(HEADS * BQ)
// WGMMA accumulator.
//
// Accumulator element 4 j + 2 h + b of thread t sits at key row
// 16 (t / 32) + (t % 32) / 4 + 8 h and query-head column 8 j + 2 (t % 4) + b.
// ``wt`` holds half that column's head weight at 2 j + b. Each (row, query)
// logit is the head sum of relu(score * key_scale) * weight, which is
// |key_scale| relu(sgn score) * weight with sgn the sign of the key scale. The
// head sums are of (score + sgn |score|) * weight / 2 = sgn relu(sgn score) *
// weight, so the stored logit is the sum times key_scale. ``sgn0`` and ``sgn1``
// are the signs at row halves 0 and 1, or 1 for a call that takes every key
// scale as nonnegative; ``key_scale`` is the scale of this thread's stored row,
// 16 (t / 32) + (t % 32) / 4 + 8 (t % 2). The four lanes that share a row
// reduce-scatter the 2 BQ sums, so lane t % 4 ends with the (query q, row half
// h) of 2 q + h congruent to it. Outside its row's window a logit is -inf when
// CLEAN; ``interior`` says the whole tile is inside every window and inside the
// tensor.
template <int HEADS, int BQ, bool CLEAN>
__device__ __forceinline__ void indexer_tile_epilogue(
    const float* acc, const float* wt, float sgn0, float sgn1, float key_scale,
    float* logits, long long off, int s0, int k0, int seq_len, int seq_len_kv,
    const int* win, bool interior, int t) {
  constexpr int N = 2 * BQ;  // (query, row half) sums a thread holds
  static_assert(N % 4 == 0, "four lanes share a row's sums");
  const int lane4 = t % 4;
  const int row = 16 * (t / 32) + (t % 32) / 4 + 8 * (lane4 % 2);
  // Sum 2 q + h of query q at row half h, two FFMAs a term; the accumulating
  // one keeps a subnormal half weight or product along the head-sum FMA chain;
  // the reduce-scatter adds and the product with the key scale still flush.
  constexpr int JQ = HEADS / 8;
  float v[N];
#pragma unroll
  for (int q = 0; q < BQ; ++q) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float sgn = h ? sgn1 : sgn0;
      const auto term = [=](float x, float w, float sum) {
        return indexer_fma(fmaf(fabsf(x), sgn, x), w, sum);
      };
      // Two chains a sum below 16 sums a thread, so at least 16 FFMA chains run
      // at once.
      float a = 0.f, b = 0.f;
#pragma unroll
      for (int jj = 0; jj < JQ; ++jj) {
        const int j = q * JQ + jj;
        const float x0 = acc[4 * j + 2 * h], x1 = acc[4 * j + 2 * h + 1];
        a = term(x0, wt[2 * j], a);
        if (N < 16) {
          b = term(x1, wt[2 * j + 1], b);
        } else {
          a = term(x1, wt[2 * j + 1], a);
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
