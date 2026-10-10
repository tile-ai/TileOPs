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

// Arrives on ``bar`` once, expecting ``bytes`` from the peers' stores.
__device__ __forceinline__ void expect_partials(void* bar, unsigned bytes) {
  unsigned m = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m),
      "r"(bytes)
      : "memory");
}

}  // namespace tileops
