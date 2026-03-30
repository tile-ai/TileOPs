from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ada_layer_norm import AdaLayerNormTest
from tests.ops.test_ada_layer_norm_zero import AdaLayerNormZeroTest
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.norm.ada_layer_norm import AdaLayerNormOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroOp

_ADA_OP_NAME = "ada_layernorm_fwd"
_ADA_ZERO_OP_NAME = "ada_layernorm_zero_fwd"


class AdaLayerNormBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        flops, _ = eval_roofline(_ADA_OP_NAME, M=t.m, N=t.n, elem_bytes=elem_bytes)
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        _, mem_bytes = eval_roofline(_ADA_OP_NAME, M=t.m, N=t.n, elem_bytes=elem_bytes)
        return mem_bytes


class AdaLayerNormZeroBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        flops, _ = eval_roofline(_ADA_ZERO_OP_NAME, M=t.m, N=t.n, elem_bytes=elem_bytes)
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        _, mem_bytes = eval_roofline(_ADA_ZERO_OP_NAME, M=t.m, N=t.n, elem_bytes=elem_bytes)
        return mem_bytes


def _manifest_params(op_name):
    params = []
    for w in load_workloads(op_name):
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype,
                                       id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype", _manifest_params(_ADA_OP_NAME))
def test_ada_layer_norm_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormTest(m, n, dtype)
    bm = AdaLayerNormBenchmark(test)
    inputs = test.gen_inputs()

    op = AdaLayerNormOp(M=m, N=n, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: PyTorch composite F.layer_norm + arithmetic
    def baseline_fn(x, scale, shift):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=test.eps)
        return scale * normed + shift

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


@pytest.mark.parametrize("m, n, dtype", _manifest_params(_ADA_ZERO_OP_NAME))
def test_ada_layer_norm_zero_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    bm = AdaLayerNormZeroBenchmark(test)
    inputs = test.gen_inputs()

    op = AdaLayerNormZeroOp(M=m, N=n, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Baseline: PyTorch composite F.layer_norm + arithmetic + gate
    def baseline_fn(x, scale, shift, gate):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=test.eps)
        return gate * (scale * normed + shift)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
