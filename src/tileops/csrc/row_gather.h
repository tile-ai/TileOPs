#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tileops {

// Stores the 16 bytes at `src` into peer CTA `peer`'s shared memory at the
// address `dst` names in this CTA, completing them on the peer's `bar`.
__device__ __forceinline__ void send16(void* dst, void* bar, int peer,
                                       const void* src) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  unsigned m = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  const unsigned* v = static_cast<const unsigned*>(src);
  asm volatile(
      "{\n\t.reg .b32 rs, rm;\n\t"
      "mapa.shared::cluster.u32 rs, %0, %2;\n\t"
      "mapa.shared::cluster.u32 rm, %1, %2;\n\t"
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [rs], {%3, "
      "%4, %5, %6}, "
      "[rm];\n\t}" ::"r"(s),
      "r"(m), "r"(peer), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3])
      : "memory");
}

// Arrives once on peer CTA `peer`'s mbarrier at the address `bar` names here.
__device__ __forceinline__ void arrive_peer(void* bar, int peer) {
  unsigned m = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n\t.reg .b32 rm;\n\t"
      "mapa.shared::cluster.u32 rm, %0, %1;\n\t"
      "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [rm];\n\t}" ::"r"(
          m),
      "r"(peer)
      : "memory");
}

// Arrives once on `bar`, expecting `bytes` of transactions from peers.
__device__ __forceinline__ void expect_bytes(void* bar, unsigned bytes) {
  unsigned m = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m),
      "r"(bytes)
      : "memory");
}

// Dequantizes 8 float8_e4m3 values at `src` by `scale` into 8 bfloat16 values
// at `dst`, rounding once to bfloat16.
__device__ __forceinline__ void dequant8(void* dst, const void* src,
                                         float scale) {
  const unsigned* in = static_cast<const unsigned*>(src);
  unsigned* out = static_cast<unsigned*>(dst);
#pragma unroll
  for (int w = 0; w < 2; ++w) {
    unsigned lo, hi;
    asm("{\n\t.reg .b16 a, b;\n\t"
        "mov.b32 {a, b}, %2;\n\t"
        "cvt.rn.f16x2.e4m3x2 %0, a;\n\t"
        "cvt.rn.f16x2.e4m3x2 %1, b;\n\t}"
        : "=r"(lo), "=r"(hi)
        : "r"(in[w]));
    float2 a = __half22float2(*reinterpret_cast<__half2*>(&lo));
    float2 b = __half22float2(*reinterpret_cast<__half2*>(&hi));
    __nv_bfloat162 pa = __floats2bfloat162_rn(a.x * scale, a.y * scale);
    __nv_bfloat162 pb = __floats2bfloat162_rn(b.x * scale, b.y * scale);
    out[2 * w] = *reinterpret_cast<unsigned*>(&pa);
    out[2 * w + 1] = *reinterpret_cast<unsigned*>(&pb);
  }
}

// 16-byte cp.async with a 256-byte L2 fetch; `live` false fills zeros.
__device__ __forceinline__ void cp_async16_l2_256(void* dst, const void* src,
                                                  bool live) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile(
      "cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;" ::"r"(s),
      "l"(src), "r"(live ? 16 : 0)
      : "memory");
}

}  // namespace tileops
