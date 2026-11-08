#pragma once

#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

namespace tileops {

// a != b, true when either is NaN, as torch.ne answers: one HSETP2.NEU.
// cutlass's operator!= is __hne, the ordered compare, which answers false
// against a NaN.
__device__ __forceinline__ bool ne_unordered(cutlass::half_t a,
                                             cutlass::half_t b) {
  return __hneu(a.to_half(), b.to_half());
}

__device__ __forceinline__ bool ne_unordered(cutlass::bfloat16_t a,
                                             cutlass::bfloat16_t b) {
  return __hneu(a.to_nv_bfloat16(), b.to_nv_bfloat16());
}

// Two 16-bit lanes of a word compared at once: each lane of the result is
// 0xffff where the compare holds and 0 where it does not. H2 is __half2 or
// __nv_bfloat162. As in torch, the ordered compares answer false against a
// NaN, and ne answers true.
template <typename H2>
__device__ __forceinline__ H2 lanes_of(unsigned word) {
  return *reinterpret_cast<const H2*>(&word);
}

template <typename H2>
__device__ __forceinline__ unsigned eq_lanes(unsigned a, unsigned b) {
  return __heq2_mask(lanes_of<H2>(a), lanes_of<H2>(b));
}

template <typename H2>
__device__ __forceinline__ unsigned ne_lanes(unsigned a, unsigned b) {
  return __hneu2_mask(lanes_of<H2>(a), lanes_of<H2>(b));
}

template <typename H2>
__device__ __forceinline__ unsigned gt_lanes(unsigned a, unsigned b) {
  return __hgt2_mask(lanes_of<H2>(a), lanes_of<H2>(b));
}

template <typename H2>
__device__ __forceinline__ unsigned lt_lanes(unsigned a, unsigned b) {
  return __hlt2_mask(lanes_of<H2>(a), lanes_of<H2>(b));
}

template <typename H2>
__device__ __forceinline__ unsigned ge_lanes(unsigned a, unsigned b) {
  return __hge2_mask(lanes_of<H2>(a), lanes_of<H2>(b));
}

template <typename H2>
__device__ __forceinline__ unsigned le_lanes(unsigned a, unsigned b) {
  return __hle2_mask(lanes_of<H2>(a), lanes_of<H2>(b));
}

// a or b nonzero: |a| equals -|b| only where both are zeros, and a NaN
// compares unequal.
template <typename H2>
__device__ __forceinline__ unsigned or_lanes(unsigned a, unsigned b) {
  return __hneu2_mask(__habs2(lanes_of<H2>(a)),
                      __hneg2(__habs2(lanes_of<H2>(b))));
}

// a and b nonzero; a NaN is nonzero.
template <typename H2>
__device__ __forceinline__ unsigned and_lanes(unsigned a, unsigned b) {
  const H2 zero = lanes_of<H2>(0u);
  return __hneu2_mask(lanes_of<H2>(a), zero) &
         __hneu2_mask(lanes_of<H2>(b), zero);
}

// Four lane masks, two a word, as four bool bytes: byte k is 1 where lane k
// holds.
__device__ __forceinline__ unsigned lane_mask_bytes(unsigned low,
                                                    unsigned high) {
  return __byte_perm(low, high, 0x6420) & 0x01010101u;
}

}  // namespace tileops
