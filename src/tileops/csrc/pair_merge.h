#pragma once

// A CTA pair merges its two halves of a WGMMA accumulator: one CTA's warpgroup
// stores its 64 x 256 float32 fragment into the peer's shared memory, and the
// peer's warpgroup of the same role weighs it and its own and adds them. Both
// warpgroups hold the same fragment layout, so a thread's 128 floats travel in
// their register order: float4 q of thread t at slot q * 128 + t, a warp's 32
// stores covering 512 contiguous bytes.

namespace tileops {

// Stores thread `t`'s 128 accumulator floats `frag` into peer CTA `peer`'s
// shared memory at the address `base` names here, completing them on the peer's
// mbarrier at the address `bar` names here.
__device__ __forceinline__ void send_frag128(const float* frag, void* base,
                                             int t, void* bar, int peer) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(base));
  unsigned m = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  unsigned rs, rm;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(rs)
               : "r"(s), "r"(peer));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(rm)
               : "r"(m), "r"(peer));
#pragma unroll
  for (int q = 0; q < 32; ++q)
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
        "[%0], {%1, %2, %3, %4}, [%5];" ::"r"(rs + 16u * (q * 128 + t)),
        "f"(frag[4 * q]), "f"(frag[4 * q + 1]), "f"(frag[4 * q + 2]),
        "f"(frag[4 * q + 3]), "r"(rm)
        : "memory");
}

// Sets thread `t`'s 128 accumulator floats `frag` to frag * mine + peer *
// theirs, the peer's floats being the ones its send_frag128 stored for `t` at
// `base`. Thread t of the warpgroup holds rows r and r + 8 of the 64 x 256
// WGMMA accumulator, r = t / 32 * 16 + t % 32 / 4; float v lies in row r when
// v % 4 < 2 and in row r + 8 otherwise. `mine` and `theirs` hold each of the 64
// rows' weight.
__device__ __forceinline__ void merge_frag128(float* frag, const void* base,
                                              int t, const float* mine,
                                              const float* theirs) {
  const float4* b = static_cast<const float4*>(base);
  const int r = t / 32 * 16 + t % 32 / 4;
  const float m0 = mine[r], m1 = mine[r + 8], t0 = theirs[r],
              t1 = theirs[r + 8];
#pragma unroll
  for (int q = 0; q < 32; ++q) {
    float4 v = b[q * 128 + t];
    frag[4 * q] = frag[4 * q] * m0 + v.x * t0;
    frag[4 * q + 1] = frag[4 * q + 1] * m0 + v.y * t0;
    frag[4 * q + 2] = frag[4 * q + 2] * m1 + v.z * t1;
    frag[4 * q + 3] = frag[4 * q + 3] * m1 + v.w * t1;
  }
}

}  // namespace tileops
