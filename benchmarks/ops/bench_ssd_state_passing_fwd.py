from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_state_passing_fwd import (
    SsdStatePassingFwdFixture,
    SsdStatePassingFwdTest,
    ssd_state_passing_fwd_ref,
)
from tileops.ops.ssd_state_passing_fwd import SsdStatePassingFwdOp


class SsdStatePassingFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        b, c, h, d = t.batch, t.num_chunks, t.n_heads, t.d_state
        # Per chunk: scale multiply + add for each (b, h, d) element
        # 2 FLOPs (mul + add) per element per chunk
        flops = b * c * h * d * 2
        return float(flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        b, c, h, d = t.batch, t.num_chunks, t.n_heads, t.d_state
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): states
        reads = b * c * h * d * elem
        # Reads (float32): dA_chunk_cumsum + initial_states
        reads += b * h * c * 4 + b * h * d * 4
        # Writes (float32): out + final_states
        writes = (b * c * h * d + b * h * d) * 4
        return float(reads + writes)


@SsdStatePassingFwdFixture
def test_ssd_state_passing_fwd_bench(batch, num_chunks, n_heads, d_state, dtype, tune):
    test = SsdStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    bm = SsdStatePassingFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdStatePassingFwdOp(batch, num_chunks, n_heads, d_state, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("ssd_state_passing_fwd", locals(), result, tag="tileops")

    def baseline(states, dA_chunk_cumsum, initial_states):
        return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("ssd_state_passing_fwd", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
