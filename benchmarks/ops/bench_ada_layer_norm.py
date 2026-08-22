import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from workloads.normalization import AdaLayerNormWorkload, AdaLayerNormZeroWorkload

_ADA_OP_NAME = "AdaLayerNormFwdOp"
_ADA_ZERO_OP_NAME = "AdaLayerNormZeroFwdOp"


def _ada_args(w: dict, dtype: torch.dtype) -> tuple:
    m, n = w["x_shape"]
    return (m, n, dtype)


@pytest.mark.parametrize("m, n, dtype", workload_params(load_workloads(_ADA_OP_NAME), _ada_args))
def test_ada_layer_norm_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = AdaLayerNormFwdOp()
    bm = ManifestBenchmark(_ADA_OP_NAME, op, test)

    # Baseline: PyTorch composite F.layer_norm + arithmetic
    def baseline_fn(x, scale, shift):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=test.eps)
        return scale * normed + shift

    bm.compare(
        {
            "tileops": op,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize(
    "m, n, dtype", workload_params(load_workloads(_ADA_ZERO_OP_NAME), _ada_args)
)
def test_ada_layer_norm_zero_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormZeroWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = AdaLayerNormZeroFwdOp()
    bm = ManifestBenchmark(_ADA_ZERO_OP_NAME, op, test)

    # Baseline: PyTorch composite F.layer_norm + arithmetic + gate
    def baseline_fn(x, scale, shift, gate):
        normed = F.layer_norm(x, (n,), weight=None, bias=None, eps=test.eps)
        return gate * (scale * normed + shift)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )
