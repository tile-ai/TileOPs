import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from workloads.fused_add_layer_norm import FusedAddLayerNormTest

_OP_NAME = "FusedAddLayerNormFwdOp"


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
def test_fused_add_layer_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddLayerNormTest(m, n, dtype)
    inputs = test.gen_inputs()

    op = FusedAddLayerNormFwdOp(dtype=dtype, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: add + F.layer_norm (separate ops)
    def baseline_fn(x, residual, weight, bias):
        add_result = (x.float() + residual.float()).to(x.dtype)
        return F.layer_norm(add_result, (n,), weight=weight, bias=bias, eps=test.eps), add_result

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
