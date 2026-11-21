#pragma once

namespace tileops {

// Stores one float from registers into peer CTA ``peer``'s ``slot``; the store
// completes its four bytes of the transaction on the peer's ``bar``.
__device__ __forceinline__ void send_partial(void* slot, void* bar, int peer,
                                             float value) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(slot));
  unsigned m = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n\t.reg .b32 rs, rm;\n\t"
      "mapa.shared::cluster.u32 rs, %0, %2;\n\t"
      "mapa.shared::cluster.u32 rm, %1, %2;\n\t"
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [rs], %3, "
      "[rm];\n\t}" ::"r"(s),
      "r"(m), "r"(peer), "f"(value)
      : "memory");
}

// Stores two floats, ``first`` then ``second``, the way send_partial
// stores one; the store completes eight bytes on the peer's ``bar``.
__device__ __forceinline__ void send_partial2(void* slot, void* bar, int peer,
                                              float first, float second) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(slot));
  unsigned m = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n\t.reg .b32 rs, rm;\n\t"
      "mapa.shared::cluster.u32 rs, %0, %2;\n\t"
      "mapa.shared::cluster.u32 rm, %1, %2;\n\t"
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [rs], {%3, "
      "%4}, [rm];\n\t}" ::"r"(s),
      "r"(m), "r"(peer), "f"(first), "f"(second)
      : "memory");
}

// Stores four floats, ``a`` to ``d``, the way send_partial stores one; the
// store completes sixteen bytes on the peer's ``bar``. ``slot`` is 16-byte
// aligned.
__device__ __forceinline__ void send_partial4(void* slot, void* bar, int peer,
                                              float a, float b, float c,
                                              float d) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(slot));
  unsigned m = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n\t.reg .b32 rs, rm;\n\t"
      "mapa.shared::cluster.u32 rs, %0, %2;\n\t"
      "mapa.shared::cluster.u32 rm, %1, %2;\n\t"
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [rs], {%3, "
      "%4, %5, %6}, [rm];\n\t}" ::"r"(s),
      "r"(m), "r"(peer), "f"(a), "f"(b), "f"(c), "f"(d)
      : "memory");
}

// Arrives on ``bar`` once, expecting ``bytes`` from the peers' stores.
__device__ __forceinline__ void expect_partials(void* bar, unsigned bytes) {
  unsigned m = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m),
      "r"(bytes)
      : "memory");
}

// Stores R floats per thread of a 128-thread warpgroup into peer CTA
// ``peer``'s ``landing`` as float4s and completes their bytes on its ``bar``.
// Thread t's i-th float4 lands at float4 index i * 128 + t, so one float4 read
// per thread is conflict-free.
template <int R>
__device__ __forceinline__ void send_partials(float const* acc, float* landing,
                                              void* bar, int peer, int t) {
#pragma unroll
  for (int i = 0; i < R / 4; ++i) {
    send_partial4(landing + (i * 128 + t) * 4, bar, peer, acc[i * 4 + 0],
                  acc[i * 4 + 1], acc[i * 4 + 2], acc[i * 4 + 3]);
  }
}

// Adds the R floats per thread that the peer's send_partials<R> stored into
// ``landing`` to ``acc``.
template <int R>
__device__ __forceinline__ void add_partials(float* acc, float const* landing,
                                             int t) {
#pragma unroll
  for (int i = 0; i < R / 4; ++i) {
    float4 const v = reinterpret_cast<float4 const*>(landing)[i * 128 + t];
    acc[i * 4 + 0] += v.x;
    acc[i * 4 + 1] += v.y;
    acc[i * 4 + 2] += v.z;
    acc[i * 4 + 3] += v.w;
  }
}

}  // namespace tileops
