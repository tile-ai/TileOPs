import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from workloads.normalization import RMSNormTest


class _RMSNormTestBaseline(RMSNormTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x_f32 / rms) * weight.float()).to(x.dtype)


_OP_NAME = "RMSNormFwdOp"


def _manifest_params():
    """Convert manifest workloads to pytest params: (m, n, dtype, tune)."""
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
def test_rms_norm_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = _RMSNormTestBaseline(m, n, dtype)
    inputs = test.gen_inputs()

    op = RMSNormFwdOp(normalized_shape=(n,), dtype=dtype, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
