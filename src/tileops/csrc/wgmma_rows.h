#pragma once

// Stores of a warpgroup's WGMMA accumulator straight to global memory.
// TileLang's fragment-to-global copy stores each thread's column pairs as
// 4-byte writes; exchanging values within each quad of lanes lets every lane
// store 16 contiguous bytes instead.

namespace tileops {

__device__ __forceinline__ unsigned pick4(unsigned a, unsigned b, unsigned c,
                                          unsigned d, int k) {
  return k == 0 ? a : k == 1 ? b : k == 2 ? c : d;
}

template <typename T>
__device__ __forceinline__ unsigned pack2(float x, float y) {
  T v[2] = {T(x), T(y)};
  return *reinterpret_cast<unsigned*>(v);
}

// One warpgroup writes its 64 x 128 f32 WGMMA accumulator `acc` into `out` (its
// first row; rows `row_stride` elements apart) in the dtype of `out`. Thread
// (warp, lane) holds rows warp * 16 + lane / 4 + {0, 8} at columns 8c + 2 (lane
// % 4) + {0, 1} as acc[4c + 2h + e]; quads exchange values so each lane stores
// 16 contiguous bytes.
template <typename T>
__device__ __forceinline__ void store_acc_rows(const float* acc, T* out,
                                               int row_stride) {
  int lane = threadIdx.x & 31;
  int q = lane & 3;
  int quad = lane & ~3;
  T* row = out + ((threadIdx.x >> 5) & 3) * 16 * row_stride +
           (lane >> 2) * row_stride;
#pragma unroll
  for (int c4 = 0; c4 < 4; ++c4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      unsigned v[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        int e = (4 * c4 + k) * 4 + 2 * h;
        v[k] = pack2<T>(acc[e], acc[e + 1]);
      }
      // Lane m of a quad holds columns 2m, 2m + 1 of group 4 c4 + k; after the
      // exchange lane q holds all eight columns of group 4 c4 + q.
      unsigned r[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        int src = (q + s) & 3;
        unsigned got =
            __shfl_sync(0xffffffffu, pick4(v[0], v[1], v[2], v[3], (q - s) & 3),
                        quad | src);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          if (m == src) r[m] = got;
        }
      }
      *reinterpret_cast<uint4*>(row + h * 8 * row_stride + (4 * c4 + q) * 8) =
          make_uint4(r[0], r[1], r[2], r[3]);
    }
  }
}

}  // namespace tileops

namespace tileops {

// One warp stores its 16 x 64 slice of a 16-bit WGMMA accumulator fragment
// (pair i at row 8 (i % 2) + lane / 4, columns 8 (i / 2) + 2 (lane % 4)) into a
// 128-byte-swizzled [rows, 64] tile at `tile` (the slice's first row) with
// four stmatrix.x4.
__device__ __forceinline__ void store_frag16_stsm(const void* frag,
                                                  void* tile) {
  const unsigned* r = static_cast<const unsigned*>(frag);
  int lane = threadIdx.x & 31;
  int k = lane >> 3;
  int row = (k & 1) * 8 + (lane & 7);
  unsigned base =
      static_cast<unsigned>(__cvta_generic_to_shared(tile)) + row * 128;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    int c = 2 * q + (k >> 1);
    unsigned addr = base + ((c ^ (lane & 7)) * 16);
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::
            "r"(addr),
        "r"(r[4 * q]), "r"(r[4 * q + 1]), "r"(r[4 * q + 2]), "r"(r[4 * q + 3])
        : "memory");
  }
}

}  // namespace tileops
