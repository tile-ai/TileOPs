from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from workloads.fused_add_rms_norm import FusedAddRMSNormTest

_OP_NAME = "FusedAddRMSNormFwdOp"


class FusedAddRMSNormBenchmark(BenchmarkBase):

    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            t = self.workload
            elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
            self._roofline_cache = eval_roofline(
                _OP_NAME, M=t.m, N=t.n, elem_bytes=elem_bytes)
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _manifest_params():
    params = []
    # Autotune has no valid configs for N=16384 (Llama-405B hidden_dim).
    _XFAIL_LABELS = {"llama-3.1-405b-prefill", "llama-3.1-405b-decode"}
    for w in load_workloads(_OP_NAME):
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            marks = ()
            if label in _XFAIL_LABELS:
                marks = pytest.mark.xfail(
                    reason="autotune has no valid configs for N=16384",
                    strict=False)
            params.append(pytest.param(m, n, dtype, True,
                                       id=f"{label}-{dtype_str}",
                                       marks=marks))
    return params


@pytest.mark.parametrize("m, n, dtype, tune", _manifest_params())
def test_fused_add_rms_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddRMSNormTest(m, n, dtype)
    bm = FusedAddRMSNormBenchmark(test)
    inputs = test.gen_inputs()

    op = FusedAddRMSNormFwdOp(M=m, N=n, dtype=dtype, tune=tune)
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
