#pragma once

namespace tl {

__device__ __forceinline__ void gqa_partial_row_sum_raw_acc_64x128(
    float* acc_s, float* row_sum) {
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    float sum = 0.0f;
#pragma unroll
    for (int value = 0; value < 32; ++value) {
      sum += acc_s[((value % 16) * 4) + (row * 2) + (value / 16)];
    }
    row_sum[row] = sum;
  }
}

}  // namespace tl
