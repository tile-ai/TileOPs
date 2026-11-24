#pragma once

namespace tileops {

// Stores one float to ``local``'s offset in the shared memory of CTA ``rank``
// of this cluster. A cluster barrier orders it before the owner's reads.
__device__ __forceinline__ void cluster_store_f32(void* local, unsigned rank,
                                                  float value) {
  unsigned remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote)
               : "r"(static_cast<unsigned>(__cvta_generic_to_shared(local))),
                 "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(value)
               : "memory");
}

}  // namespace tileops
