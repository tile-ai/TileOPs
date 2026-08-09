// Helper for TileLang scatter kernel: atomicAdd with pointer offset.
// Needed because TileLang's T.atomic_add(buf[dynamic_idx], 1, return_prev=True)
// does not support dynamic indices on global memory (codegen limitation).
#pragma once
__device__ __forceinline__ int tl_atomic_add_offset(int* base, int offset, int val) {
    return atomicAdd(base + offset, val);
}
