#pragma once

// Decode, WGMMA and promotion helpers of the MXFP4 M-grouped GEMM. The packed
// e2m1 weights become e4m3 register fragments of an fp8 WGMMA whose other
// operand is the fp8 rows of ``a`` as TMA loads them.

#include <cuda.h>
#include <cuda_bf16.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

#include "moe_grouped_gemm_fp8_helper.h"

namespace tileops {

// The float an e8m0 code stands for: 2 ** (code - 127), 2 ** -127 at code 0
// and NaN at 255, as torch's float8_e8m0fnu reads them.
__device__ __forceinline__ float mxfp4_e8m0_value(uint32_t code) {
  uint32_t const bits =
      code == 0 ? 0x00400000u : (code == 255 ? 0x7fc00000u : code << 23);
  return __uint_as_float(bits);
}

// Binades a 128-wide K step's four e8m0 weight scales of a row may span and
// still fold exactly: e2m1's smallest value 0.5 times 2 ** -5 is e4m3's
// smallest normal, so lowering e4m3 exponents by up to 5 keeps every value.
constexpr uint32_t kMxfp4FoldSpan = 5;

// ``x`` as m * 2 ** e with m in [1, 2) (zero, infinities and NaN as
// themselves with e = 0), by bit operations: --use_fast_math flushes subnormal
// operands of float arithmetic, and an e8m0 scale of code 0 is 2 ** -127.
__device__ __forceinline__ int mxfp4_split(float x, float* mantissa) {
  uint32_t const bits = __float_as_uint(x);
  uint32_t const exponent = (bits >> 23) & 0xffu;
  if (exponent == 0xffu || (bits & 0x7fffffffu) == 0) {
    *mantissa = x;
    return 0;
  }
  if (exponent == 0) {
    uint32_t const fraction = bits & 0x7fffffu;
    int const shift = __clz(fraction) - 8;
    *mantissa = __uint_as_float((bits & 0x80000000u) | 0x3f800000u |
                                ((fraction << shift) & 0x7fffffu));
    return -126 - shift;
  }
  *mantissa = __uint_as_float((bits & 0x807fffffu) | 0x3f800000u);
  return static_cast<int>(exponent) - 127;
}

// value * row * token with the scales' product never formed on its own: the
// mantissas multiply (into [1, 4)) and the exponents add, so nothing
// overflows or flushes before the result does.
__device__ __forceinline__ float mxfp4_scaled_exact(float value, float row,
                                                    float token) {
  float row_mantissa;
  float token_mantissa;
  int const exponent =
      mxfp4_split(row, &row_mantissa) + mxfp4_split(token, &token_mantissa);
  return ldexpf(value * (row_mantissa * token_mantissa), exponent);
}

// Scales whose every pairwise product is a finite, normal float32.
constexpr float kMxfp4ScaleLow = 1.0842021724855044e-19f;  // 2 ** -63
constexpr float kMxfp4ScaleHigh = 9.2233720368547758e18f;  // 2 ** 63

__device__ __forceinline__ bool mxfp4_scale_fast(float x) {
  float const magnitude = fabsf(x);
  return magnitude >= kMxfp4ScaleLow && magnitude <= kMxfp4ScaleHigh;
}

// e8m0 codes kMxfp4CodeLow..kMxfp4CodeHigh: 2 ** -63 .. 2 ** 63.
constexpr uint32_t kMxfp4CodeLow = 64;
constexpr uint32_t kMxfp4CodeHigh = 190;

__device__ __forceinline__ bool mxfp4_code_fast(uint32_t code) {
  return code >= kMxfp4CodeLow && code <= kMxfp4CodeHigh;
}

// How a K step's weights enter the contraction.
enum Mxfp4Mode : int {
  // e8m0: each sub-block scaled by 2 ** (b - max) over its row's step,
  // exact for spans up to kMxfp4FoldSpan; ``scales`` gets 2 ** max.
  kMxfp4Fold = 0,
  // e8m0: unscaled; ``scales`` gets each sub-block's 2 ** b.
  kMxfp4SplitE8M0 = 1,
  // float32: unscaled; ``scales`` gets each sub-block's scale.
  kMxfp4SplitF32 = 2,
};

// The e4m3 bytes of the e2m1 magnitudes {0, .5, 1, 1.5} and {2, 3, 4, 6}: the
// byte row a code's low three bits select from. Adding d times kMxfp4BinadeLo
// and kMxfp4BinadeHi moves every nonzero byte's exponent by d binades, which
// scales it by 2 ** d exactly while d lies in [-5, 6]: 0.5 * 2 ** -5 is e4m3's
// smallest normal and 6 * 2 ** 6 = 384 lies below its largest finite value.
constexpr uint32_t kMxfp4MagnitudesLo = 0x3c383000u;
constexpr uint32_t kMxfp4MagnitudesHi = 0x4c484440u;
constexpr uint32_t kMxfp4BinadeLo = 0x08080800u;
constexpr uint32_t kMxfp4BinadeHi = 0x08080808u;

// Four e2m1 codes (``codes`` low 16 bits, k order, sign in bit 3) as e4m3 bytes
// in k order: magnitudes from the byte row ``row`` (8 bytes), signs from the
// codes. A selector with bit 3 set fills its byte with the selected byte's top
// bit, so ``signs`` is 0xff where a code is negative and 0x80 elsewhere; bit 6
// tells them apart.
__device__ __forceinline__ uint32_t mxfp4_e2m1x4_e4m3(uint32_t codes, uint2 row,
                                                      uint32_t fill) {
  uint32_t const magnitude = __byte_perm(row.x, row.y, codes & 0x7777u);
  // prmt itself: __byte_perm's runtime selector drops the sign-fill bit.
  // ``fill`` is 0x80808080.
  uint32_t signs;
  asm("prmt.b32 %0, %1, %1, %2;" : "=r"(signs) : "r"(fill), "r"(codes));
  return magnitude | ((signs << 1) & 0x80808080u);
}

// 16-byte chunk ``chunk`` (K steps 4 chunk .. 4 chunk + 3) of row ``row`` of
// a task's code table that comes by TMA: Rows-row boxes of BoxBytes (64 or 128)
// of each row, TMA-swizzled (16-byte chunk c of a row at byte offset o stored
// at c ^ ((o >> 7) & (BoxBytes / 16 - 1))).
template <int BoxBytes, int Rows>
__device__ __forceinline__ uint8_t const* mxfp4_table_chunk(void const* table,
                                                            int row,
                                                            int chunk) {
  int const box = chunk * 16 / BoxBytes;
  int const row_offset = row * BoxBytes;
  int const swizzled =
      (chunk % (BoxBytes / 16)) ^ ((row_offset >> 7) & (BoxBytes / 16 - 1));
  return static_cast<uint8_t const*>(table) + box * Rows * BoxBytes +
         row_offset + swizzled * 16;
}

// Word ``kk`` (four e8m0 codes of K step ``kk``) of row ``row`` of a task's
// code table: mxfp4_table_chunk's layout under BoxBytes > 0, otherwise rows of
// Steps words back to back.
template <int Steps, int BoxBytes, int Rows>
__device__ __forceinline__ uint32_t mxfp4_table_word(void const* table, int row,
                                                     int kk) {
  if constexpr (BoxBytes > 0) {
    return *reinterpret_cast<uint32_t const*>(
        mxfp4_table_chunk<BoxBytes, Rows>(table, row, kk >> 2) + (kk & 3) * 4);
  } else {
    return static_cast<uint32_t const*>(table)[row * Steps + kk];
  }
}

// The e4m3 register A fragments of a 64-row warp-group's weights for one
// 128-wide K step: rows g = 16 * (warp % 4) + lane / 4 and g + 8 of the
// warp-group's rows from ``row0`` of the packed stage (rows of 64 bytes as a
// 64B-swizzled TMA copy lays them out: 16-byte chunk c of row r at chunk
// c ^ (r / 2 % 4), so the eight rows a warp reads hit distinct banks); K32
// slice i (the i-th 32-wide sub-block) in ``frags[4 i .. 4 i + 3]`` as an fp8
// WGMMA A operand holds it: (g, k 4q .. 4q + 3), (g + 8, same), (g, k 4q + 16
// .. 4q + 19), (g + 8, same) with q = lane % 4. Byte j of a row holds k = 2j
// (low nibble) and 2j + 1. ``scales`` gets the rows' scales as ``Mode`` says:
// rows g's then g + 8's, one each under kMxfp4Fold, four each otherwise.
template <int Mode, int Steps = 0, int BoxBytes = 0, int Rows = 0>
__device__ __forceinline__ void mxfp4_weight_step(void const* packed_stage,
                                                  void const* scale_stage,
                                                  int row0, uint32_t* frags,
                                                  float* scales, int kk = 0) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const g =
      row0 + ((static_cast<int>(threadIdx.x) >> 5) & 3) * 16 + lane / 4;
  // Bytes 2q and 2q + 1 of each 8-byte half of a sub-block's 16 bytes.
  int const offset = 2 * (lane & 3);
  // Rows g and g + 8 share their chunk swizzle.
  int const swizzle = (g >> 1) & 3;
  uint8_t const* rows[2] = {
      static_cast<uint8_t const*>(packed_stage) + g * 64 + offset,
      static_cast<uint8_t const*>(packed_stage) + (g + 8) * 64 + offset};
  uint2 folds[2][4];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    if constexpr (Mode == kMxfp4SplitF32) {
      float4 const f = static_cast<float4 const*>(scale_stage)[g + 8 * r];
      scales[4 * r] = f.x;
      scales[4 * r + 1] = f.y;
      scales[4 * r + 2] = f.z;
      scales[4 * r + 3] = f.w;
    } else {
      // A ring stage's scales (a word a row), or the task's code table's.
      uint32_t const word =
          Steps > 0 ? mxfp4_table_word<Steps, BoxBytes, Rows>(scale_stage,
                                                              g + 8 * r, kk)
                    : static_cast<uint32_t const*>(scale_stage)[g + 8 * r];
      uint32_t codes[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        codes[i] = __byte_perm(word, 0, 0x4440u + i);
      }
      if constexpr (Mode == kMxfp4Fold) {
        uint32_t const top =
            max(max(codes[0], codes[1]), max(codes[2], codes[3]));
        scales[r] = mxfp4_e8m0_value(top);
        uint32_t const lo = kMxfp4MagnitudesLo - top * kMxfp4BinadeLo;
        uint32_t const hi = kMxfp4MagnitudesHi - top * kMxfp4BinadeHi;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          folds[r][i] = make_uint2(lo + codes[i] * kMxfp4BinadeLo,
                                   hi + codes[i] * kMxfp4BinadeHi);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          scales[4 * r + i] = mxfp4_e8m0_value(codes[i]);
        }
      }
    }
    if constexpr (Mode != kMxfp4Fold) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        folds[r][i] = make_uint2(kMxfp4MagnitudesLo, kMxfp4MagnitudesHi);
      }
    }
  }
  uint32_t const fill = 0x80808080u;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      uint8_t const* chunk = rows[r] + 16 * (i ^ swizzle);
      uint32_t const lo = *reinterpret_cast<uint16_t const*>(chunk);
      uint32_t const hi = *reinterpret_cast<uint16_t const*>(chunk + 8);
      frags[4 * i + r] = mxfp4_e2m1x4_e4m3(lo, folds[r][i], fill);
      frags[4 * i + 2 + r] = mxfp4_e2m1x4_e4m3(hi, folds[r][i], fill);
    }
  }
}

// Folds a word of four e8m0 codes (a K step of a row) into running maxima:
// the largest span of a step, the largest and the smallest code.
__device__ __forceinline__ void mxfp4_scan_word(uint32_t word, uint32_t& span,
                                                uint32_t& top, uint32_t& low) {
  uint32_t const c0 = __byte_perm(word, 0, 0x4440u);
  uint32_t const c1 = __byte_perm(word, 0, 0x4441u);
  uint32_t const c2 = __byte_perm(word, 0, 0x4442u);
  uint32_t const c3 = __byte_perm(word, 0, 0x4443u);
  uint32_t const step_top = max(max(c0, c1), max(c2, c3));
  uint32_t const step_low = min(min(c0, c1), min(c2, c3));
  span = max(span, step_top - step_low);
  top = max(top, step_top);
  low = min(low, step_low);
}

// Whether every 128-wide K step of the thread's weight rows g and g + 8 (as
// mxfp4_weight_step picks them, from row ``first`` of ``scales``) takes the
// fast path: every scale within mxfp4_scale_fast's range and, under e8m0
// scales, a step's four spanning at most kMxfp4FoldSpan binades (so they
// fold exactly). ``scales`` holds rows of Steps groups of four scales; rows
// past ``last`` read it. The four lanes of a row split its steps, their loads
// all issued before any is used; a lane past the last step rereads step 0.
template <int Steps, bool E8M0>
__device__ __forceinline__ bool mxfp4_rows_fast(void const* scales, int first,
                                                int last) {
  int const g = first + ((static_cast<int>(threadIdx.x) >> 5) & 3) * 16 +
                (static_cast<int>(threadIdx.x) & 31) / 4;
  int const lane_step = static_cast<int>(threadIdx.x) & 3;
  constexpr int kPerLane = (Steps + 3) / 4;
  int64_t const rows[2] = {static_cast<int64_t>(min(g, last)) * Steps,
                           static_cast<int64_t>(min(g + 8, last)) * Steps};
  bool fast = true;
  if constexpr (E8M0) {
    uint32_t words[2][kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        int const kk = 4 * j + lane_step;
        words[r][j] = static_cast<uint32_t const*>(
            scales)[rows[r] + (kk < Steps ? kk : 0)];
      }
    }
    uint32_t span = 0, top = 0, low = 255;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        mxfp4_scan_word(words[r][j], span, top, low);
      }
    }
    fast =
        span <= kMxfp4FoldSpan && mxfp4_code_fast(top) && mxfp4_code_fast(low);
  } else {
    float4 words[2][kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        int const kk = 4 * j + lane_step;
        words[r][j] =
            static_cast<float4 const*>(scales)[rows[r] + (kk < Steps ? kk : 0)];
      }
    }
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float4 const f = words[r][j];
        fast = fast && mxfp4_scale_fast(f.x) && mxfp4_scale_fast(f.y) &&
               mxfp4_scale_fast(f.z) && mxfp4_scale_fast(f.w);
      }
    }
  }
  return fast;
}

// Arrives on ``barrier`` expecting ``bytes`` and copies them from ``src`` to
// ``dst`` by one bulk copy completing on it; both 16-byte aligned, ``bytes`` a
// multiple of 16.
__device__ __forceinline__ void mxfp4_bulk_load(void* dst, void const* src,
                                                void* barrier, uint32_t bytes) {
  uint32_t const bar = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
      "r"(bytes)
      : "memory");
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], "
      "%2, [%3];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}

// mxfp4_rows_fast under e8m0 scales, of the tile's codes in shared memory:
// mxfp4_table_word's table, rows past ``last`` reading it; the thread's rows
// those of mxfp4_weight_step from row ``wg_row0``. The four lanes of a row
// split its steps: under the swizzled table in 16-byte chunks of four steps,
// otherwise a word at a time, a lane past the last step rereading step 0.
template <int Steps, int BoxBytes, int Rows>
__device__ __forceinline__ bool mxfp4_rows_fast_shared(void const* codes,
                                                       int wg_row0, int last) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const g =
      wg_row0 + ((static_cast<int>(threadIdx.x) >> 5) & 3) * 16 + lane / 4;
  int const lane_step = lane & 3;
  uint32_t span = 0, top = 0, low = 255;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int const row = min(g + 8 * r, last);
    if constexpr (BoxBytes > 0) {
      // Steps is a multiple of 16: each lane takes Steps / 16 chunks of a row.
#pragma unroll
      for (int j = 0; j < Steps / 16; ++j) {
        uint4 const words = *reinterpret_cast<uint4 const*>(
            mxfp4_table_chunk<BoxBytes, Rows>(codes, row, 4 * j + lane_step));
        mxfp4_scan_word(words.x, span, top, low);
        mxfp4_scan_word(words.y, span, top, low);
        mxfp4_scan_word(words.z, span, top, low);
        mxfp4_scan_word(words.w, span, top, low);
      }
    } else {
#pragma unroll
      for (int j = 0; j < (Steps + 3) / 4; ++j) {
        int const kk = 4 * j + lane_step;
        mxfp4_scan_word(mxfp4_table_word<Steps, BoxBytes, Rows>(
                            codes, row, kk < Steps ? kk : 0),
                        span, top, low);
      }
    }
  }
  return span <= kMxfp4FoldSpan && mxfp4_code_fast(top) && mxfp4_code_fast(low);
}

// Whether ``value`` holds for every one of ``count`` threads at named barrier
// ``id``, which they all reach.
__device__ __forceinline__ int mxfp4_all_threads(int value, int id, int count) {
  int result;
  asm volatile(
      "{\n .reg .pred p, q;\n setp.ne.s32 p, %1, 0;\n"
      " barrier.red.and.pred q, %2, %3, p;\n selp.s32 %0, 1, 0, q;\n}"
      : "=r"(result)
      : "r"(value), "r"(id), "r"(count)
      : "memory");
  return result;
}

// K32 slices [First, First + Count) of a warp-group's 64 x N x 128 fp8
// product, the weights from registers (``frags``, four per slice) and the
// tokens from a 128B-swizzled tile of 128-byte rows at ``b_lo``. Slice i adds
// to accumulator set i % Sets of ``accumulator`` (Sets sets of N / 2 floats),
// so Sets chains of WGMMAs run independently; each set is overwritten at its
// first slice.
template <int N, int First, int Count, int Sets>
__device__ __forceinline__ void mxfp4_wgmma_rs(float* accumulator,
                                               uint32_t* frags, uint32_t b_lo) {
  constexpr int kRegs = N / 2;
#pragma unroll
  for (int set = 0; set < Sets; ++set) {
    tl::warpgroup_fence_operand(accumulator + set * kRegs, kRegs);
  }
#pragma unroll
  for (int ki = First; ki < First + Count; ++ki) {
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(frags + 4 * ki), 4);
  }
  tl::warpgroup_arrive();
#pragma unroll
  for (int ki = First; ki < First + Count; ++ki) {
    tl::wgmma_rs<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3,
                 tl::DataType::kFloat32, 64, N, 32, false, false, 1, 1>(
        frags + 4 * ki, kMoeFp8WgmmaDescHi | uint64_t(b_lo + ki * 2),
        reinterpret_cast<uint32_t*>(accumulator + (ki % Sets) * kRegs),
        ki - First >= Sets);
  }
  tl::warpgroup_commit_batch();
#pragma unroll
  for (int ki = First; ki < First + Count; ++ki) {
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(frags + 4 * ki), 4);
  }
#pragma unroll
  for (int set = 0; set < Sets; ++set) {
    tl::warpgroup_fence_operand(accumulator + set * kRegs, kRegs);
  }
}

// The warp-group's 64 x N accumulator of weight rows by tokens, in bfloat16,
// transposed into the staging tile of the output: token t,
// column c of the CTA's tile at box c / 64, row t, 16-byte chunk
// (c % 64) / 8 ^ (t % 8), as a 128B-swizzled TMA store box of 64 columns and
// N rows reads it. ``col0`` is the warp-group's first column.
template <int N>
__device__ __forceinline__ void mxfp4_stage_transposed(float* accumulator,
                                                       void* staging,
                                                       int col0) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const c0 =
      col0 + ((static_cast<int>(threadIdx.x) >> 5) & 3) * 16 + lane / 4;
  int const pair = (lane & 3) * 2;
  uint8_t* base = static_cast<uint8_t*>(staging);
#pragma unroll
  for (int i = 0; i < N / 8; ++i) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int const c = c0 + (e >= 2 ? 8 : 0);
      int const t = i * 8 + pair + (e & 1);
      __nv_bfloat16 const v = __float2bfloat16_rn(accumulator[i * 4 + e]);
      uint8_t* dst = base + (c / 64) * N * 128 + t * 128 +
                     ((((c % 64) / 8) ^ (t % 8)) * 16) + (c % 8) * 2;
      *reinterpret_cast<__nv_bfloat16*>(dst) = v;
    }
  }
}

// The warp's token scales of a step into its shared slice: lane l's
// ``values[j]`` is token l + 32 j's, for the N tokens of the tile. Only the
// warp reads the slice, so __syncwarp orders the stores against the reads of
// this and of the slice's previous step.
template <int N>
__device__ __forceinline__ void mxfp4_stage_tokens(float* slice,
                                                   float const* values) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  __syncwarp();
#pragma unroll
  for (int j = 0; j < (N + 31) / 32; ++j) {
    if (lane + 32 * j < N) {
      slice[lane + 32 * j] = values[j];
    }
  }
  __syncwarp();
}

// ---------------------------------------------------------------------------
// Shared-memory decode: the weights decoded once per CTA into an fp8 tile that
// is the N operand of an SS WGMMA whose M operand is the tokens. A task's
// weight row folds every sub-block of every K step to one reference code,
// ``ref``: its e4m3 bytes carry 2 ** (code - ref), and the row's 2 ** (ref -
// 127) multiplies its column once after the last step.
// ---------------------------------------------------------------------------

// Binades a row's reference lies under its largest code: the largest e2m1
// value, 6, raised six binades is 384, under e4m3's largest finite 448.
constexpr uint32_t kMxfp4FoldRise = 6;
// A row's codes may span this many binades and still fold to one reference:
// its codes lie in [ref - kMxfp4FoldSpan, ref + kMxfp4FoldRise].
constexpr uint32_t kMxfp4RowSpan = kMxfp4FoldSpan + kMxfp4FoldRise;

// Whether a task's rows fold, and each weight row's reference. Decode thread
// ``thread`` of ``Threads`` scans weight rows ``thread + Threads j`` and token
// rows ``Threads - 1 - thread + Threads j``, so the two kinds of row fall to
// different threads. A weight row past ``last`` (past N) reads row ``last``.
// A weight row folds when its codes lie in [kMxfp4CodeLow, kMxfp4CodeHigh]
// (2 ** -63 .. 2 ** 63, so no product of a step under its token scale
// overflows or flushes) and span at most kMxfp4RowSpan binades; its reference,
// top - kMxfp4FoldRise, goes to ``refs[row]`` and 2 ** (ref - 127) to
// ``column_scales[row]``. A token row folds when each of its codes is 0
// (2 ** -127, which the promotion flushes to zero, as the register-fragment
// path does) or lies in [kMxfp4CodeLow, kMxfp4CodeHigh]. ``table`` is
// mxfp4_table_chunk's; ``tokens`` holds rows of Steps codes back to back.
template <int Steps, int BoxBytes, int BlockN, int BlockM, int Threads>
__device__ __forceinline__ bool mxfp4_scan_task(void const* table,
                                                uint8_t const* tokens,
                                                int* refs, float* column_scales,
                                                int last, int thread) {
  static_assert(BoxBytes > 0, "the code table comes by TMA in boxes");
  bool fast = true;
#pragma unroll
  for (int j = 0; j < (BlockN + Threads - 1) / Threads; ++j) {
    int const row = thread + Threads * j;
    if (row < BlockN) {
      int const source = min(row, last);
      // Bytewise running maxima and minima of the row's codes.
      uint32_t tops = 0, lows = 0xffffffffu;
#pragma unroll
      for (int c = 0; c < Steps / 4; ++c) {
        uint4 const words = *reinterpret_cast<uint4 const*>(
            mxfp4_table_chunk<BoxBytes, BlockN>(table, source, c));
        tops = __vmaxu4(__vmaxu4(tops, words.x),
                        __vmaxu4(words.y, __vmaxu4(words.z, words.w)));
        lows = __vminu4(__vminu4(lows, words.x),
                        __vminu4(words.y, __vminu4(words.z, words.w)));
      }
      uint32_t const top = max(max(tops & 0xffu, (tops >> 8) & 0xffu),
                               max((tops >> 16) & 0xffu, tops >> 24));
      uint32_t const low = min(min(lows & 0xffu, (lows >> 8) & 0xffu),
                               min((lows >> 16) & 0xffu, lows >> 24));
      fast = fast && top - low <= kMxfp4RowSpan && mxfp4_code_fast(top) &&
             mxfp4_code_fast(low);
      int const ref = static_cast<int>(top - kMxfp4FoldRise);
      refs[row] = ref;
      column_scales[row] =
          __uint_as_float(static_cast<uint32_t>(max(ref, 1)) << 23);
    }
  }
#pragma unroll
  for (int j = 0; j < (BlockM + Threads - 1) / Threads; ++j) {
    int const row = Threads - 1 - thread + Threads * j;
    if (row < BlockM) {
#pragma unroll
      for (int c = 0; c < Steps / 16; ++c) {
        uint4 const words =
            reinterpret_cast<uint4 const*>(tokens + row * Steps)[c];
        uint32_t const w[4] = {words.x, words.y, words.z, words.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          uint32_t const in_range =
              __vcmpgeu4(w[i], kMxfp4CodeLow * 0x01010101u) &
              __vcmpleu4(w[i], kMxfp4CodeHigh * 0x01010101u);
          fast = fast && (in_range | __vcmpeq4(w[i], 0u)) == 0xffffffffu;
        }
      }
    }
  }
  return fast;
}

// Decode thread ``thread``'s references: those of the weight rows its units
// cover (a unit is sub-block ``unit % 4`` of row ``unit / 4``): whole units
// ``thread + Threads j``, then the unit of its half, mxfp4_decode_shared's.
// Threads at or past kHalves take no half; they store a clamped reference in
// slot kWhole that they never read.
template <int BlockN, int Threads>
__device__ __forceinline__ void mxfp4_unit_refs(int const* refs, int* unit_refs,
                                                int thread) {
  constexpr int kWhole = 4 * BlockN / Threads;
#pragma unroll
  for (int j = 0; j < kWhole; ++j) {
    unit_refs[j] = refs[(thread + Threads * j) / 4];
  }
  if (4 * BlockN % Threads != 0) {
    unit_refs[kWhole] =
        refs[min((kWhole * Threads + thread / 2) / 4, BlockN - 1)];
  }
}

// One arrival of the calling warp on ``barrier`` once every lane is here.
__device__ __forceinline__ void mxfp4_warp_arrive(void* barrier) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0) {
    uint32_t const bar =
        static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar)
                 : "memory");
  }
}

// The magnitude byte rows (mxfp4_e2m1x4_e4m3's ``row``) of decode thread
// ``thread``'s units for K step ``kk``, two words a unit in
// mxfp4_decode_shared's order: under ``fold`` each sub-block's rows carry
// 2 ** (code - ref) of its row's reference (``unit_refs``); otherwise none. The
// codes come from mxfp4_table_word's table, resident for the whole task, so
// the rows are built before the step's stages are waited for.
template <int Steps, int BoxBytes, int BlockN, int Threads>
__device__ __forceinline__ void mxfp4_decode_luts(void const* table,
                                                  int const* unit_refs,
                                                  bool fold, int kk, int thread,
                                                  uint32_t* luts) {
  constexpr int kUnits = 4 * BlockN;
  constexpr int kWhole = kUnits / Threads;
  constexpr int kHalves = 2 * (kUnits - kWhole * Threads);
#pragma unroll
  for (int j = 0; j <= kWhole; ++j) {
    if (j < kWhole || (kHalves > 0 && thread < kHalves)) {
      int const unit =
          j < kWhole ? thread + Threads * j : kWhole * Threads + thread / 2;
      uint32_t const code = __byte_perm(
          mxfp4_table_word<Steps, BoxBytes, BlockN>(table, unit >> 2, kk), 0,
          0x4440u + (unit & 3));
      uint32_t const shift =
          fold ? code - static_cast<uint32_t>(unit_refs[j]) : 0u;
      luts[2 * j] = kMxfp4MagnitudesLo + shift * kMxfp4BinadeLo;
      luts[2 * j + 1] = kMxfp4MagnitudesHi + shift * kMxfp4BinadeHi;
    }
  }
}

// A K step's weights, decoded by thread ``thread`` into the stage's fp8 tile:
// rows of 128 bytes under the 128B swizzle an SS WGMMA descriptor reads
// (16-byte chunk c of row r at chunk c ^ (r % 8)), from the packed stage's rows
// of 64 bytes, which the calling warp releases on ``packed_release`` once it
// has converted them; ``luts`` from mxfp4_decode_luts. Each thread takes the
// same number of whole units and the units left over go out as 16-value halves
// to threads [0, kHalves). With 512 units on 224 threads, warps 0-3 take 2.5
// units a lane and warps 4-6 take 2.
template <int BlockN, int Threads>
__device__ __forceinline__ void mxfp4_decode_shared(void const* packed_stage,
                                                    void* decoded_stage,
                                                    void* packed_release,
                                                    uint32_t const* luts,
                                                    int thread) {
  constexpr int kUnits = 4 * BlockN;
  constexpr int kWhole = kUnits / Threads;
  constexpr int kHalves = 2 * (kUnits - kWhole * Threads);
  static_assert(kHalves <= Threads, "a thread takes at most one half unit");
  uint4 packed[kWhole];
  uint2 half_packed;
#pragma unroll
  for (int j = 0; j < kWhole; ++j) {
    packed[j] = *reinterpret_cast<uint4 const*>(
        static_cast<uint8_t const*>(packed_stage) +
        (thread + Threads * j) * 16);
  }
  bool const takes_half = kHalves > 0 && thread < kHalves;
  if (takes_half) {
    half_packed = *reinterpret_cast<uint2 const*>(
        static_cast<uint8_t const*>(packed_stage) + kWhole * Threads * 16 +
        thread * 8);
  }
  // 0x80808080 (``thread`` < 2 ** 31) held in a per-thread register: as a
  // uniform constant, NVCC rebuilds it with a copy before each of the eight
  // prmts that read it a unit.
  uint32_t const fill = 0x80808080u | (static_cast<uint32_t>(thread) >> 31);
#pragma unroll
  for (int j = 0; j < kWhole; ++j) {
    int const unit = thread + Threads * j;
    int const row = unit >> 2;
    int const sub_block = unit & 3;
    uint2 const lut = make_uint2(luts[2 * j], luts[2 * j + 1]);
    // The sub-block's 32 e2m1 codes (byte i holding k = 2i in its low nibble
    // and 2i + 1 in its high one) as 32 e4m3 bytes in k order.
    uint32_t const words[4] = {packed[j].x, packed[j].y, packed[j].z,
                               packed[j].w};
    uint32_t out[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[2 * i] = mxfp4_e2m1x4_e4m3(words[i], lut, fill);
      out[2 * i + 1] = mxfp4_e2m1x4_e4m3(words[i] >> 16, lut, fill);
    }
    uint8_t* dst = static_cast<uint8_t*>(decoded_stage) + row * 128;
    int const swizzle = row & 7;
    *reinterpret_cast<uint4*>(dst + (((2 * sub_block) ^ swizzle) << 4)) =
        make_uint4(out[0], out[1], out[2], out[3]);
    *reinterpret_cast<uint4*>(dst + (((2 * sub_block + 1) ^ swizzle) << 4)) =
        make_uint4(out[4], out[5], out[6], out[7]);
  }
  if (takes_half) {
    int const unit = kWhole * Threads + thread / 2;
    int const half = thread & 1;
    int const row = unit >> 2;
    uint2 const lut = make_uint2(luts[2 * kWhole], luts[2 * kWhole + 1]);
    uint32_t const out[4] = {mxfp4_e2m1x4_e4m3(half_packed.x, lut, fill),
                             mxfp4_e2m1x4_e4m3(half_packed.x >> 16, lut, fill),
                             mxfp4_e2m1x4_e4m3(half_packed.y, lut, fill),
                             mxfp4_e2m1x4_e4m3(half_packed.y >> 16, lut, fill)};
    uint8_t* dst = static_cast<uint8_t*>(decoded_stage) + row * 128;
    *reinterpret_cast<uint4*>(dst +
                              (((2 * (unit & 3) + half) ^ (row & 7)) << 4)) =
        make_uint4(out[0], out[1], out[2], out[3]);
  }
  mxfp4_warp_arrive(packed_release);
}

// K32 slice ``slice`` of a warp-group's 64 x BlockN x 128 fp8 product,
// overwriting ``accumulator``; ``a_lo`` and ``b_lo`` are 128B-swizzled K-major
// descriptor low words of the step's tiles.
template <int BlockN>
__device__ __forceinline__ void mxfp4_wgmma_ss_slice(float* accumulator,
                                                     uint32_t a_lo,
                                                     uint32_t b_lo, int slice) {
  tl::warpgroup_fence_operand(accumulator, BlockN / 2);
  tl::warpgroup_arrive();
  tl::wgmma_ss<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3,
               tl::DataType::kFloat32, 64, BlockN, 32, false, false, 1, 1>(
      kMoeFp8WgmmaDescHi | uint64_t(a_lo + slice * 2),
      kMoeFp8WgmmaDescHi | uint64_t(b_lo + slice * 2),
      reinterpret_cast<uint32_t*>(accumulator), 0);
  tl::warpgroup_commit_batch();
  tl::warpgroup_fence_operand(accumulator, BlockN / 2);
}

// final += step * weight scale * token scale over a warp-group's 64 x BlockN
// SS accumulator for sub-block ``sub_block`` of K step ``kk``, each product by
// mxfp4_scaled_exact: the thread's rows take ``token0`` and ``token1``, its
// columns (weight rows) their codes from mxfp4_table_word's table.
template <int Steps, int BoxBytes, int BlockN>
__device__ __forceinline__ void mxfp4_promote_exact(float const* step,
                                                    float* final_accum,
                                                    void const* table, int kk,
                                                    int sub_block, float token0,
                                                    float token1) {
  int const pair = (static_cast<int>(threadIdx.x) & 3) * 2;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float const weight = mxfp4_e8m0_value(
          __byte_perm(mxfp4_table_word<Steps, BoxBytes, BlockN>(
                          table, i * 8 + pair + e, kk),
                      0, 0x4440u + sub_block));
      final_accum[i * 4 + e] +=
          mxfp4_scaled_exact(step[i * 4 + e], weight, token0);
      final_accum[i * 4 + 2 + e] +=
          mxfp4_scaled_exact(step[i * 4 + 2 + e], weight, token1);
    }
  }
}

// A warp-group's 64 x BlockN SS accumulator times each column's scale.
template <int BlockN>
__device__ __forceinline__ void mxfp4_scale_columns(
    float* final_accum, float const* column_scales) {
  int const pair = (static_cast<int>(threadIdx.x) & 3) * 2;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    float2 const scale =
        *reinterpret_cast<float2 const*>(column_scales + i * 8 + pair);
    final_accum[i * 4 + 0] *= scale.x;
    final_accum[i * 4 + 1] *= scale.y;
    final_accum[i * 4 + 2] *= scale.x;
    final_accum[i * 4 + 3] *= scale.y;
  }
}

// The calling thread's shared stores, made visible to the async proxy (an SS
// WGMMA reads them), then its warp's arrival on ``barrier``.
__device__ __forceinline__ void mxfp4_publish_stage(void* barrier) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  mxfp4_warp_arrive(barrier);
}

// The token codes of a math thread's accumulator rows ``row0`` and
// ``row0 + 8`` for K steps ``kk`` .. ``kk + 3`` (``kk`` a multiple of 4), a
// byte a step, from ``tokens``' rows of ``steps`` codes back to back
// (``steps`` a multiple of 16).
__device__ __forceinline__ void mxfp4_token_words(uint8_t const* tokens,
                                                  int row0, int steps, int kk,
                                                  uint32_t* words) {
  words[0] = *reinterpret_cast<uint32_t const*>(tokens + row0 * steps + kk);
  words[1] =
      *reinterpret_cast<uint32_t const*>(tokens + (row0 + 8) * steps + kk);
}

// The two token scales of a math thread's accumulator rows at step ``j`` of
// mxfp4_token_words' words. Under a folded task, whose codes are 0 or in [64,
// 190], ``code << 23`` as a float: 2 ** (code - 127), and 0.0 at code 0, which
// the promotion treats like mxfp4_e8m0_value's 2 ** -127 because
// --use_fast_math flushes that subnormal to zero. mxfp4_e8m0_value otherwise.
__device__ __forceinline__ void mxfp4_token_scales(uint32_t const* words, int j,
                                                   bool fold, float* out) {
  uint32_t const c0 = (words[0] >> (8 * j)) & 0xffu;
  uint32_t const c1 = (words[1] >> (8 * j)) & 0xffu;
  if (fold) {
    out[0] = __uint_as_float(c0 << 23);
    out[1] = __uint_as_float(c1 << 23);
  } else {
    out[0] = mxfp4_e8m0_value(c0);
    out[1] = mxfp4_e8m0_value(c1);
  }
}

// A task's fold flag, which the math warp-groups read once the task's first
// decoded stage is published.
__device__ __forceinline__ void mxfp4_store_flag(int* flag, int value) {
  *reinterpret_cast<volatile int*>(flag) = value;
}

}  // namespace tileops
