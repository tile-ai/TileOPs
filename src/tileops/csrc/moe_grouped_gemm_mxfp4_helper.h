#pragma once

// Decode, WGMMA and promotion helpers of the MXFP4 M-grouped GEMM. The packed
// e2m1 weights become e4m3 register fragments of an fp8 WGMMA whose other
// operand is the fp8 rows of ``a`` as TMA loads them.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

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

// e8m0 codes 64..190: 2 ** -63 .. 2 ** 63.
__device__ __forceinline__ bool mxfp4_code_fast(uint32_t code) {
  return code >= 64u && code <= 190u;
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
// Word ``kk`` (four e8m0 codes of K step ``kk``) of row ``row`` of a task's
// code table: under BoxBytes > 0, Rows-row boxes of BoxBytes (64 or 128) of
// each row, TMA-swizzled (16-byte chunk c of a row at byte offset o stored at c
// ^ ((o >> 7) & (BoxBytes / 16 - 1))); otherwise rows of Steps words back to
// back.
template <int Steps, int BoxBytes, int Rows>
__device__ __forceinline__ uint32_t mxfp4_table_word(void const* table, int row,
                                                     int kk) {
  uint8_t const* const base = static_cast<uint8_t const*>(table);
  if constexpr (BoxBytes > 0) {
    int const byte = kk * 4;
    int const box = byte / BoxBytes;
    int const in_box = byte % BoxBytes;
    int const row_offset = row * BoxBytes;
    int const chunk = (in_box / 16) ^ ((row_offset >> 7) & (BoxBytes / 16 - 1));
    return *reinterpret_cast<uint32_t const*>(
        base + box * Rows * BoxBytes + row_offset + chunk * 16 + in_box % 16);
  } else {
    return reinterpret_cast<uint32_t const*>(base)[row * Steps + kk];
  }
}

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
  // e4m3 {0, .5, 1, 1.5} and {2, 3, 4, 6}. Lowering every nonzero exponent by
  // d scales them by 2 ** -d exactly while d <= kMxfp4FoldSpan.
  constexpr uint32_t kRowLo = 0x3c383000u, kRowHi = 0x4c484440u;
  constexpr uint32_t kStepLo = 0x08080800u, kStepHi = 0x08080808u;
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
        uint32_t const lo = kRowLo - top * kStepLo;
        uint32_t const hi = kRowHi - top * kStepHi;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          folds[r][i] =
              make_uint2(lo + codes[i] * kStepLo, hi + codes[i] * kStepHi);
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
        folds[r][i] = make_uint2(kRowLo, kRowHi);
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
      uint8_t const* const base =
          static_cast<uint8_t const*>(codes) + row * BoxBytes;
      int const swizzle = ((row * BoxBytes) >> 7) & (BoxBytes / 16 - 1);
#pragma unroll
      for (int j = 0; j < Steps / 16; ++j) {
        int const chunk = 4 * j + lane_step;  // 16-byte chunk of the row
        int const box = chunk * 16 / BoxBytes;
        int const in_box = chunk % (BoxBytes / 16);
        uint4 const words = *reinterpret_cast<uint4 const*>(
            base + box * Rows * BoxBytes + ((in_box ^ swizzle) * 16));
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
  constexpr uint64_t kDescHi = uint64_t(0x40000040u) << 32;
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
        frags + 4 * ki, kDescHi | uint64_t(b_lo + ki * 2),
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

}  // namespace tileops
