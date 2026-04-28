// Helpers to force ptxas to treat wgmma.wait_group as a scheduling barrier.
//
// wait_wgmma_anchor<N>(sink, cond) fuses `wgmma.wait_group.sync.aligned N`
// with a guarded shared-memory store. The store is a real side effect
// ptxas cannot eliminate; the guard predicate depends on a runtime counter
// ptxas cannot constant-fold. Result: an uncrossable barrier that anchors
// softmax between wait<1> and wait<0>.

#pragma once

#include <cstdint>

namespace tl {

__device__ __forceinline__ void tileops_barrier_arrive_named(int barrier_id,
                                                             int thread_count) {
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(thread_count));
}

template <int N>
__device__ __forceinline__ void wait_wgmma_anchor(int* sink_smem_ptr,
                                                   int cond) {
  unsigned smem_u32 =
      static_cast<unsigned>(__cvta_generic_to_shared(sink_smem_ptr));
  asm volatile(
      "{\n"
      "  wgmma.wait_group.sync.aligned %0;\n"
      "  .reg .pred p_anc_%=;\n"
      "  setp.eq.s32 p_anc_%=, %2, 0xdeadbeef;\n"
      "  @p_anc_%= st.shared.u32 [%1], %2;\n"
      "}\n" ::"n"(N),
      "r"(smem_u32), "r"(cond)
      : "memory");
}

}  // namespace tl
