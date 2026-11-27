"""Benchmarks for the fused gated elementwise ops, one case per manifest call.

Every row times the reference in torch eager and through inductor, and flashinfer's
kernel, which takes the concatenated input the reference splits. A clamped row also times
vLLM's ``SiluAndMulWithClamp`` kernel, which scales no row.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    baseline_module,
    compiled_reference,
    flashinfer_op,
)
from tileops._ops.elementwise import (
    GELUAndMulFwdOp,
    SiLUAndMulFwdOp,
)


def _vllm_silu_and_mul_with_clamp(x: torch.Tensor, limit: float):
    """vLLM's ``SiluAndMulWithClamp`` kernel, into an output allocated outside timing."""
    config = baseline_module("vllm.config", VLLM_TAG)
    activation = baseline_module("vllm.model_executor.layers.activation", VLLM_TAG)
    with config.set_current_vllm_config(config.VllmConfig()):
        layer = activation.SiluAndMulWithClamp(limit)
    out = x.new_empty(x.shape[0], x.shape[1] // 2)

    def run(x, *_absent):
        layer.op(out, x, limit, 1.0, 0.0)
        return out

    return run


# flashinfer names its fused gated kernels after the same three activations and
# takes the same concatenated input, so a key here doubles as its entry-point name.
def _profile_fused_gated(op_cls, case: bench.Case, library: str) -> None:
    op = op_cls(**case.arguments)
    flashinfer_fn = flashinfer_op(library)
    implementations = {}
    limit = case.arguments.get("swiglu_limit", 0.0)
    scaled = any(t is not None for t in case.inputs[1:])
    if limit > 0 or scaled:
        print(f"  [skip] {FLASHINFER_TAG}: {library} clamps no input and scales no row")
    else:
        implementations[FLASHINFER_TAG] = lambda x, *_absent: flashinfer_fn(x)
    if limit > 0:
        implementations[VLLM_TAG] = bench.Implementation(
            run=_vllm_silu_and_mul_with_clamp(case.inputs[0], limit),
            noncomparable_reason=(
                "scales no row: vLLM applies the routing weight in its down-projection GEMM"
                if scaled
                else None
            ),
        )
    implementations["torch-ref"] = case.torch
    implementations[TORCH_COMPILE_TAG] = compiled_reference(case.torch)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SiLUAndMulFwdOp))
def test_silu_and_mul_bench(case) -> None:
    _profile_fused_gated(SiLUAndMulFwdOp, case, "silu_and_mul")


@pytest.mark.parametrize("case", bench.cases(GELUAndMulFwdOp))
def test_gelu_and_mul_bench(case) -> None:
    tanh = case.arguments.get("approximate") == "tanh"
    _profile_fused_gated(GELUAndMulFwdOp, case, "gelu_tanh_and_mul" if tanh else "gelu_and_mul")
