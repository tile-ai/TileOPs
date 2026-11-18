#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <tl_templates/cuda/common.h>

namespace tileops {

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

// Convert the 16 e4m3 bytes of `raw` and store them as 16-byte chunks
// `lo_chunk` and `lo_chunk + 1` of the 128-byte row at `dst_row`, chunks
// swizzled by row j % 8.
template <typename OutT>
TL_DEVICE void fp8x16_to_wide_store(uint4 raw, char* dst_row, int lo_chunk,
                                    int j) {
  uint4 const lo_words =
      make_uint4(fp8x2_to_wide<OutT>(raw.x), fp8x2_to_wide<OutT>(raw.x >> 16),
                 fp8x2_to_wide<OutT>(raw.y), fp8x2_to_wide<OutT>(raw.y >> 16));
  uint4 const hi_words =
      make_uint4(fp8x2_to_wide<OutT>(raw.z), fp8x2_to_wide<OutT>(raw.z >> 16),
                 fp8x2_to_wide<OutT>(raw.w), fp8x2_to_wide<OutT>(raw.w >> 16));
  *reinterpret_cast<uint4*>(dst_row + ((lo_chunk ^ (j & 7)) << 4)) = lo_words;
  *reinterpret_cast<uint4*>(dst_row + (((lo_chunk + 1) ^ (j & 7)) << 4)) =
      hi_words;
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
    // Columns 16c .. 16c + 15: 16-bit chunks 2c % 8 and 2c % 8 + 1 of half
    // c / 4.
    fp8x16_to_wide_store<OutT>(raw, half_out + row, 2 * (c & 3), j);
  }
}

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

// fp8_tile_to_wide_sw128 in the same layouts, for a thread that can hold its
// whole share of the tile: every load is issued before the first conversion,
// so the loads' latencies overlap instead of each stalling its conversion.
template <int ROWS, int THREADS, typename OutT>
TL_DEVICE void fp8_tile_to_wide_sw128_prefetched(const void* src, void* dst,
                                                 int t, int valid_rows) {
  static_assert(sizeof(OutT) == 2, "the converted tile is 16-bit");
  static_assert(ROWS % (THREADS / 8) == 0,
                "every thread converts as many rows");
  constexpr int STEP = THREADS / 8;
  constexpr int ITERS = ROWS / STEP;
  const char* in = static_cast<const char*>(src);
  int const c = t & 7;
  char* const half_out = static_cast<char*>(dst) + (c >> 2) * ROWS * 128;
  uint4 raw[ITERS];
#pragma unroll
  for (int k = 0; k < ITERS; ++k) {
    int const j = (t >> 3) + k * STEP;
    raw[k] = *reinterpret_cast<const uint4*>(
        in + (j >> 3) * 1024 + (j & 7) * 128 + ((c ^ (j & 7)) << 4));
  }
#pragma unroll
  for (int k = 0; k < ITERS; ++k) {
    int const j = (t >> 3) + k * STEP;
    uint4 const r = j < valid_rows ? raw[k] : make_uint4(0, 0, 0, 0);
    fp8x16_to_wide_store<OutT>(r, half_out + (j >> 3) * 1024 + (j & 7) * 128,
                               2 * (c & 3), j);
  }
}

// fp8_tile_to_wide_sw128 for a tile of ROWS x 64: the e4m3 rows are 64 bytes
// swizzled at 64 bytes, row j, column d at byte
//   (j / 8) * 512 + (j % 8) * 64 + ((d / 16) ^ (j / 2 % 4)) * 16 + d % 16,
// and the 16-bit rows 128 bytes swizzled at 128 bytes, element
//   (j / 8) * 512 + (j % 8) * 64 + ((d / 8) ^ (j % 8)) * 8 + d % 8.
// Thread t of THREADS (a multiple of 4) converts column chunk t % 4 of rows
// t / 4, t / 4 + THREADS / 4, ...; a row at or past `valid_rows` is written as
// zero.
template <int ROWS, int THREADS, typename OutT>
TL_DEVICE void fp8_tile_to_wide_d64(const void* src, void* dst, int t,
                                    int valid_rows) {
  static_assert(sizeof(OutT) == 2, "the converted tile is 16-bit");
  static_assert(THREADS % 4 == 0, "a thread keeps one 16-byte column chunk");
  const char* in = static_cast<const char*>(src);
  char* out = static_cast<char*>(dst);
  int const c = t & 3;
#pragma unroll 4
  for (int j = t >> 2; j < ROWS; j += THREADS / 4) {
    uint4 raw = *reinterpret_cast<const uint4*>(
        in + (j >> 3) * 512 + (j & 7) * 64 + ((c ^ ((j >> 1) & 3)) << 4));
    if (j >= valid_rows) {
      raw = make_uint4(0, 0, 0, 0);
    }
    fp8x16_to_wide_store<OutT>(raw, out + (j >> 3) * 1024 + (j & 7) * 128,
                               2 * c, j);
  }
}

}  // namespace tileops
