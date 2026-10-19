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
// Thread t of `threads` takes 16-byte chunks t, t + threads, ...; a row at or
// past `valid_rows` is written as zero, so a stale page row cannot reach the
// value product.
template <int ROWS, typename OutT>
TL_DEVICE void fp8_tile_to_wide_sw128(const void* src, void* dst, int t,
                                      int threads, int valid_rows) {
  static_assert(sizeof(OutT) == 2, "the converted tile is 16-bit");
  const char* in = static_cast<const char*>(src);
  char* out = static_cast<char*>(dst);
#pragma unroll 4
  for (int e = t; e < ROWS * 8; e += threads) {
    int const j = e >> 3;
    int const c = e & 7;
    int const row = (j >> 3) * 1024 + (j & 7) * 128;
    uint4 raw =
        *reinterpret_cast<const uint4*>(in + row + ((c ^ (j & 7)) << 4));
    if (j >= valid_rows) {
      raw = make_uint4(0, 0, 0, 0);
    }
    uint32_t const words[4] = {raw.x, raw.y, raw.z, raw.w};
    uint32_t wide[8];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        __nv_fp8x2_storage_t const pair =
            static_cast<__nv_fp8x2_storage_t>(words[w] >> (16 * h));
        __half2_raw const f16 = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
        if constexpr (std::is_same_v<OutT, half_t>) {
          wide[2 * w + h] = static_cast<uint32_t>(f16.x) |
                            (static_cast<uint32_t>(f16.y) << 16);
        } else {
          float2 const f32 =
              __half22float2(*reinterpret_cast<const __half2*>(&f16));
          __nv_bfloat162 const bf = __float22bfloat162_rn(f32);
          wide[2 * w + h] = *reinterpret_cast<const uint32_t*>(&bf);
        }
      }
    }
    // Columns 16c .. 16c + 15: 16-bit chunks 2c % 8 and 2c % 8 + 1 of half c
    // / 4.
    char* half_base =
        out + (c >> 2) * ROWS * 128 + (j >> 3) * 1024 + (j & 7) * 128;
    int const lo = (2 * (c & 3)) ^ (j & 7);
    int const hi = (2 * (c & 3) + 1) ^ (j & 7);
    *reinterpret_cast<uint4*>(half_base + (lo << 4)) =
        make_uint4(wide[0], wide[1], wide[2], wide[3]);
    *reinterpret_cast<uint4*>(half_base + (hi << 4)) =
        make_uint4(wide[4], wide[5], wide[6], wide[7]);
  }
}

}  // namespace tileops
