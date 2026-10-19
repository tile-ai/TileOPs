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

}  // namespace tileops
