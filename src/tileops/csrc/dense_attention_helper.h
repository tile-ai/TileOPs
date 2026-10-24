#pragma once

#include <tl_templates/cuda/barrier.h>

namespace tileops {

__device__ __forceinline__ void dense_attention_wait(Barrier* barriers,
                                                     int index, int phase) {
  if (!barriers[index].try_wait(phase)) {
    barriers[index].wait(phase);
  }
}

}  // namespace tileops
