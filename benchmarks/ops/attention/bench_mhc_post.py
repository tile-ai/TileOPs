from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MHCPostOp
from workloads.ops.mhc_post import MHCPostTest


class _MHCPostTestBaseline(MHCPostTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
            batch, n_expand * c_x) + x_res.float()
        x_out_ref = x_out_ref.bfloat16()
        return x_out_ref


class MHCPostBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        flops = 2 * t.batch * (
            t.n_expand * t.n_expand * t.c_x * t.c_x + t.n_expand * t.c_x)
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return (t.n_expand * 2 + 1) * t.c_x


_MHC_POST_BENCH_PARAMS = [
    pytest.param(1, 4, 1280, torch.bfloat16, True, id="small"),
    pytest.param(2, 4, 1920, torch.bfloat16, True, id="medium"),
    pytest.param(4, 4, 2560, torch.bfloat16, True, id="large"),
]


@pytest.mark.parametrize("batch, n_expand, c_x, dtype, tune", _MHC_POST_BENCH_PARAMS)
def test_mhc_post_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                         tune: bool) -> None:
    test = _MHCPostTestBaseline(batch, n_expand, c_x, dtype)
    bm = MHCPostBenchmark(test)
    inputs = test.gen_inputs()

    op = MHCPostOp(batch, n_expand, c_x, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
