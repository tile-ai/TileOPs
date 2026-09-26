"""Benchmarks for AdaLayerNorm and AdaLayerNormZero, each case one manifest call.

No library ships either kernel, so every row is torch against itself, eager and compiled.
"""

import pytest
import torch.nn.functional as F

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from workloads.normalization import NormCall


def _bench(op_cls: type, call, baseline_fn) -> None:
    workload = NormCall(call)
    inputs = workload.gen_inputs()
    op = op_cls(**workload.arguments())
    assert_matches_reference(op, baseline_fn, *inputs, **reference_tolerance(inputs[0].dtype))
    ManifestBenchmark(op, workload).compare(
        {
            "tileops": op,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(AdaLayerNormFwdOp))
def test_ada_layer_norm_bench(call) -> None:
    eps = call.params["eps"]

    def baseline_fn(x, scale, shift):
        normed = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
        return (scale.float() * normed + shift.float()).to(x.dtype)

    _bench(AdaLayerNormFwdOp, call, baseline_fn)


@pytest.mark.parametrize("call", manifest_calls(AdaLayerNormZeroFwdOp))
def test_ada_layer_norm_zero_bench(call) -> None:
    eps = call.params["eps"]

    def baseline_fn(x, scale, shift, gate):
        normed = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
        return (gate.float() * (scale.float() * normed + shift.float())).to(x.dtype)

    _bench(AdaLayerNormZeroFwdOp, call, baseline_fn)
