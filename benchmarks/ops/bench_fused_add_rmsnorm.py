from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_fused_add_rmsnorm import FusedAddRmsNormTest
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.norm.fused_add_rmsnorm import FusedAddRmsNormOp

_OP_NAME = "fused_add_rmsnorm_fwd"


class FusedAddRmsNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        flops, _ = eval_roofline(_OP_NAME, M=t.m, N=t.n, elem_bytes=elem_bytes)
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        _, mem_bytes = eval_roofline(_OP_NAME, M=t.m, N=t.n, elem_bytes=elem_bytes)
        return mem_bytes


def _manifest_params():
    params = []
    for w in load_workloads(_OP_NAME):
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype, True,
                                       id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype, tune", _manifest_params())
def test_fused_add_rmsnorm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddRmsNormTest(m, n, dtype)
    bm = FusedAddRmsNormBenchmark(test)
    inputs = test.gen_inputs()

    op = FusedAddRmsNormOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: add + manual rmsnorm (separate ops)
    def baseline_fn(x, residual, weight):
        add_result = (x.float() + residual.float()).to(x.dtype)
        rms = torch.sqrt(add_result.float().pow(2).mean(dim=-1, keepdim=True) + test.eps)
        y = ((add_result.float() / rms) * weight.float()).to(x.dtype)
        return y, add_result

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
