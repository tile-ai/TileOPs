#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <tl_templates/cuda/common.h>

namespace tileops {

// Convert one e4m3 key or value tile of ROWS x 128 into float16 or bfloat16,
// shared memory to shared memory, in the layouts TileLang's
// make_swizzled_layout gives the two buffers. Row j, column d of the e4m3 tile
// is byte
//   (j / 8) * 1024 + (j % 8) * 128 + ((d / 16) ^ (j % 8)) * 16 + d % 16,
// and of the 16-bit tile element
//   (d / 64) * ROWS * 64 + (j / 8) * 512 + (j % 8) * 64
//     + (((d % 64) / 8) ^ (j % 8)) * 8 + d % 8.
// Thread t of THREADS (a multiple of 8) converts column chunk t % 8 of rows
// t / 8, t / 8 + THREADS / 8, ...; a row at or past `valid_rows` is written as
// zero, so a stale page row cannot reach the value product.
// The 16-bit pair that e4m3 bytes b0 (low) and b1 convert to, packed as one
// word.
template <typename OutT>
TL_DEVICE uint32_t fp8x2_to_wide(uint32_t bytes) {
  __half2_raw const f16 = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(bytes), __NV_E4M3);
  if constexpr (std::is_same_v<OutT, half_t>) {
    return *reinterpret_cast<const uint32_t*>(&f16);
  } else {
    __nv_bfloat162 const bf = __float22bfloat162_rn(
        __half22float2(*reinterpret_cast<const __half2*>(&f16)));
    return *reinterpret_cast<const uint32_t*>(&bf);
  }
}

template <int ROWS, int THREADS, typename OutT, bool CUT>
TL_DEVICE void fp8_tile_to_wide_rows(const char* in, char* out, int t,
                                     int valid_rows) {
  // THREADS is a multiple of 8, so a thread keeps one column chunk and steps
  // down THREADS / 8 rows at a time.
  int const c = t & 7;
  char* const half_out = out + (c >> 2) * ROWS * 128;
#pragma unroll 4
  for (int j = t >> 3; j < ROWS; j += THREADS / 8) {
    int const row = (j >> 3) * 1024 + (j & 7) * 128;
    uint4 raw =
        *reinterpret_cast<const uint4*>(in + row + ((c ^ (j & 7)) << 4));
    if constexpr (CUT) {
      if (j >= valid_rows) {
        raw = make_uint4(0, 0, 0, 0);
      }
    }
    uint4 const lo_words = make_uint4(
        fp8x2_to_wide<OutT>(raw.x), fp8x2_to_wide<OutT>(raw.x >> 16),
        fp8x2_to_wide<OutT>(raw.y), fp8x2_to_wide<OutT>(raw.y >> 16));
    uint4 const hi_words = make_uint4(
        fp8x2_to_wide<OutT>(raw.z), fp8x2_to_wide<OutT>(raw.z >> 16),
        fp8x2_to_wide<OutT>(raw.w), fp8x2_to_wide<OutT>(raw.w >> 16));
    // Columns 16c .. 16c + 15: 16-bit chunks 2c % 8 and 2c % 8 + 1 of half
    // c / 4.
    char* const dst = half_out + row;
    *reinterpret_cast<uint4*>(dst + (((2 * (c & 3)) ^ (j & 7)) << 4)) =
        lo_words;
    *reinterpret_cast<uint4*>(dst + (((2 * (c & 3) + 1) ^ (j & 7)) << 4)) =
        hi_words;
  }
}

template <int ROWS, int THREADS, typename OutT>
TL_DEVICE void fp8_tile_to_wide_sw128(const void* src, void* dst, int t,
                                      int valid_rows) {
  static_assert(sizeof(OutT) == 2, "the converted tile is 16-bit");
  static_assert(THREADS % 8 == 0, "a thread keeps one 16-byte column chunk");
  const char* in = static_cast<const char*>(src);
  char* out = static_cast<char*>(dst);
  if (valid_rows >= ROWS) {
    fp8_tile_to_wide_rows<ROWS, THREADS, OutT, false>(in, out, t, valid_rows);
  } else {
    fp8_tile_to_wide_rows<ROWS, THREADS, OutT, true>(in, out, t, valid_rows);
  }
}

// An e4m3 value product reads V transposed: e4m3 WGMMA reads B K-major only.
// Within each 16 keys, key slot k of the transposed tile holds key
//   2 * (k / 4) + k % 2 + 8 * ((k / 2) % 2),
// the key whose probability the score accumulator already holds in the thread
// that reads slot k of the e4m3 A fragment, so rounding P to that fragment
// exchanges nothing between lanes (fp8_pack_probs below).
//
// Transpose one e4m3 value tile of ROWS x 128, ROWS a multiple of 16, into the
// 128 x 128 K-major tile of those key slots, both in TileLang's 128-byte
// swizzle: row r, byte c is (r / 8) * 1024 + (r % 8) * 128
// + ((c / 16) ^ (r % 8)) * 16 + c % 16. Slots past ROWS are left as they are,
// and a key at or past `valid_rows` is written as zero.
//
// A warp moves 16 keys x 32 channels a step, read as b16 pairs of channels: a
// transposing ldmatrix of keys 0-7 and 8-15 of two 16-channel chunks gives
// lane 4c + t keys 2t, 2t + 1 and 2t + 8, 2t + 9 of channel pairs c, which one
// byte permute splits into the four slots 4t .. 4t + 3 of a channel, and
// stmatrix writes each channel's 16 slots as one row.
template <int ROWS, int THREADS, bool CUT>
TL_DEVICE void fp8_tile_transpose_chunks(const char* in, char* out, int t,
                                         int valid_rows) {
  constexpr int WARPS = THREADS / 32;
  int const warp = t / 32;
  int const lane = t % 32;
  // Matrix m = lane / 8 of a step: keys 8 * (m % 2) onward of the step's
  // channel chunk 2s + m / 2; this lane gives the address of its row r. The
  // read row is key 8 * (m % 2) + r of the 16, the stored one channel
  // 16 * (2s + m / 2) + 2r + m % 2; both swizzles fold into per-lane offsets.
  int const m = lane >> 3;
  int const r = lane & 7;
  int const dl = (2 * r + (m & 1)) & 7;  // stored row % 8
  uint32_t const in_lane = static_cast<uint32_t>(__cvta_generic_to_shared(in)) +
                           (m & 1) * 1024 + r * 128;
  uint32_t const out_lane =
      static_cast<uint32_t>(__cvta_generic_to_shared(out)) +
      ((m >> 1) * 2 + (r >> 2)) * 1024 + dl * 128;
  int const t2 = 2 * (lane & 3);
#pragma unroll 1
  for (int chunk = warp; chunk < ROWS / 16; chunk += WARPS) {
    uint32_t const in_chunk = in_lane + chunk * 2048;
    uint32_t const out_chunk = out_lane + ((chunk ^ dl) << 4);
    uint32_t lo = 0xffffffffu;
    uint32_t hi = 0xffffffffu;
    if constexpr (CUT) {
      // v0, v2 hold keys 2t, 2t + 1 of the chunk; v1, v3 keys 2t + 8, 2t + 9.
      int const k = chunk * 16 + t2;
      lo = (k >= valid_rows ? 0u : 0xffffu) |
           (k + 1 >= valid_rows ? 0u : 0xffff0000u);
      hi = (k + 8 >= valid_rows ? 0u : 0xffffu) |
           (k + 9 >= valid_rows ? 0u : 0xffff0000u);
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t v0, v1, v2, v3;
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, "
          "[%4];"
          : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
          : "r"(in_chunk + (((2 * s + (m >> 1)) ^ r) << 4)));
      if constexpr (CUT) {
        v0 &= lo;
        v2 &= lo;
        v1 &= hi;
        v3 &= hi;
      }
      // Byte 2e + b of v holds channel 2c + b of key 2t + e (+ 8 in v1, v3).
      uint32_t const s0 = __byte_perm(v0, v1, 0x6420);
      uint32_t const s1 = __byte_perm(v0, v1, 0x7531);
      uint32_t const s2 = __byte_perm(v2, v3, 0x6420);
      uint32_t const s3 = __byte_perm(v2, v3, 0x7531);
      asm volatile(
          "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
          :
          : "r"(out_chunk + s * 4096), "r"(s0), "r"(s1), "r"(s2), "r"(s3)
          : "memory");
    }
  }
}

template <int ROWS, int THREADS>
TL_DEVICE void fp8_tile_transpose_sw128(const void* src, void* dst, int t,
                                        int valid_rows) {
  static_assert(ROWS % 16 == 0 && ROWS <= 128,
                "16-key chunks of one 128-byte row");
  static_assert(THREADS % 32 == 0, "whole warps");
  const char* in = static_cast<const char*>(src);
  char* out = static_cast<char*>(dst);
  if (valid_rows >= ROWS) {
    fp8_tile_transpose_chunks<ROWS, THREADS, false>(in, out, t, valid_rows);
  } else {
    fp8_tile_transpose_chunks<ROWS, THREADS, true>(in, out, t, valid_rows);
  }
}

// Four floats rounded to e4m3, the first in the low byte.
TL_DEVICE uint32_t f32x4_to_e4m3x4(float a, float b, float c, float d) {
  uint32_t out;
  // One asm block, so the two halves merge into one register in place.
  asm("{\n"
      ".reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}"
      : "=r"(out)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return out;
}

// Round a 64 x BLOCK_N score accumulator to the e4m3 A fragment of the value
// product, slot k taking the key the transpose puts at k. Register 4g + e of a
// 32-key step holds key 8g + 2 * (lane % 4) + e % 2 of row lane / 4 + 8 * (e /
// 2); fragment word 2h + r of the step holds slots 16h + 4 * (lane % 4) .. + 3
// of row lane / 4 + 8r. A tile of a multiple of 16 keys ends in a half step.
template <int BLOCK_N>
TL_DEVICE void fp8_pack_probs(const float* acc, void* frag) {
  static_assert(BLOCK_N % 16 == 0, "16-key chunks");
  uint32_t* p = static_cast<uint32_t*>(frag);
#pragma unroll
  for (int h = 0; h < BLOCK_N / 16; ++h) {
    const float* a = acc + 8 * h;
    p[2 * h + 0] = f32x4_to_e4m3x4(a[0], a[1], a[4], a[5]);
    p[2 * h + 1] = f32x4_to_e4m3x4(a[2], a[3], a[6], a[7]);
  }
}

}  // namespace tileops
