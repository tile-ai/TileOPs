from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_decode import (
    SsdDecodeFixture,
    SsdDecodeTest,
    ssd_decode_ref,
)
from tileops.ops.ssd_decode import SsdDecodeOp


class SsdDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        b, h, p, n = t.batch, t.n_heads, t.d_head, t.d_state
        # State update: dA * old_s + dt * x * B  -> 3 muls + 1 add per (b,h,p,n)
        # Output accum: new_s * C                 -> 1 mul + 1 add per (b,h,p,n)
        # Total: 6 * b * h * p * n
        return float(6 * b * h * p * n)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        b, h, p, n, g = t.batch, t.n_heads, t.d_head, t.d_state, t.n_groups
        f32, dtype_bytes = 4, 2  # float32 and float16/bfloat16
        # Reads: A(h) + dt(b,h) + x(b,h,p) + B_in(b,g,n) + C_in(b,g,n) + state(b,h,p,n)
        reads = (
            h * f32
            + b * h * f32
            + b * h * p * dtype_bytes
            + 2 * b * g * n * dtype_bytes
            + b * h * p * n * f32
        )
        # Writes: state(b,h,p,n) + y_out(b,h,p)
        writes = (b * h * p * n + b * h * p) * f32
        return float(reads + writes)


@SsdDecodeFixture
def test_ssd_decode_bench(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SsdDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    bm = SsdDecodeBenchmark(test)
    A, dt, x, B_in, C_in, state = test.gen_inputs()

    op = SsdDecodeOp(batch, n_heads, d_head, d_state, n_groups, dtype, tune=tune)
    result = bm.profile(op, A, dt, x, B_in, C_in, state)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    state_bl = state.clone()

    def baseline(A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)

    result_bl = bm.profile(baseline, A, dt, x, B_in, C_in, state_bl)
    BenchmarkReport.record("ssd_decode", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
