#pragma once

// ptxas places a key step's wgmma.wait_group for the value product right after
// the row-max reduction, ahead of every exponential, so the consumer idles on
// the product and then computes all of them. A shared store of a few
// probabilities, issued before the wait, keeps their exponentials ahead of it:
// ptxas keeps a shared store in order with a following wgmma.wait_group and
// cannot move a computation feeding the store past it.

namespace tileops {

// Where a CUDA post-processing pass puts value_wait_anchor. It has no
// definition, so a slot the pass leaves in place fails to link instead of
// silently dropping the anchor.
__device__ void value_wait_anchor_slot(void*);

// Store `value` to the calling thread's word of `scratch`.
__device__ __forceinline__ void value_wait_anchor(float* scratch, float value) {
  unsigned addr = static_cast<unsigned>(
      __cvta_generic_to_shared(scratch + (threadIdx.x & 127)));
  asm volatile("st.shared.f32 [%0], %1;" ::"r"(addr), "f"(value) : "memory");
}

}  // namespace tileops
