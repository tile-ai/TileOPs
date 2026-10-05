"""Benchmarks for the fused gated elementwise ops, one case per manifest call.

Every row times the reference in torch eager and through inductor, and flashinfer's
kernel, which takes the concatenated input the reference splits. A second test checks
that each kernel's default strategy is the fast one.
"""

import pytest
import torch

from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flashinfer_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.kernels.elementwise import (
    GeluAndMulFwdKernel,
    GeluTanhAndMulFwdKernel,
    SiluAndMulFwdKernel,
)
from tileops.ops.elementwise import (
    GeluAndMulFwdOp,
    GeluTanhAndMulFwdOp,
    SiluAndMulFwdOp,
)
from workloads.elementwise import (
    ElementwiseCall,
)


# flashinfer names its fused gated kernels after the same three activations and
# takes the same concatenated input, so a key here doubles as its entry-point name.
def _profile_fused_gated(op_cls, call, library: str) -> None:
    workload = ElementwiseCall(call)
    op = op_cls(**workload.arguments())
    inputs = workload.gen_inputs()
    flashinfer_fn = flashinfer_op(library)
    ManifestBenchmark(op, workload).compare(
        {
            "tileops": op,
            FLASHINFER_TAG: flashinfer_fn,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(SiluAndMulFwdOp))
def test_silu_and_mul_bench(call) -> None:
    _profile_fused_gated(SiluAndMulFwdOp, call, "silu_and_mul")


@pytest.mark.parametrize("call", manifest_calls(GeluAndMulFwdOp))
def test_gelu_and_mul_bench(call) -> None:
    _profile_fused_gated(GeluAndMulFwdOp, call, "gelu_and_mul")


@pytest.mark.parametrize("call", manifest_calls(GeluTanhAndMulFwdOp))
def test_gelu_tanh_and_mul_bench(call) -> None:
    _profile_fused_gated(GeluTanhAndMulFwdOp, call, "gelu_tanh_and_mul")


def _strategy_params():
    """Default-strategy sentinel: shape and dtype axes on the first kernel, plus
    one reference-point direct-vs-explicit sentinel per remaining kernel.

    The three ops share the fused-gated wrapper but bind different activation
    bodies, whose instruction and register cost can flip the direct-vs-explicit
    result — so each kernel keeps a sentinel, without re-sweeping shapes.
    """
    # Scenario -> (tokens, width).
    strategy_shapes = {
        "llama-hidden-1k-tokens": (1024, 4096),
        "llama-7b-ffn-1k-tokens": (1024, 11008),
        "llama-hidden-4k-tokens": (4096, 4096),
    }
    strategy_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    strategy_kernels = [
        ("silu_and_mul", SiluAndMulFwdKernel),
        ("gelu_and_mul", GeluAndMulFwdKernel),
        ("gelu_tanh_and_mul", GeluTanhAndMulFwdKernel),
    ]
    (sweep_op, sweep_cls), sentinels = strategy_kernels[0], strategy_kernels[1:]
    ref_scenario, ref_dtype = next(iter(strategy_shapes)), torch.float16

    def case(op_name, scenario, dtype, kernel_cls, mark):
        case_id = f"{op_name}-{scenario}-{str(dtype).removeprefix('torch.')}"
        M, N = strategy_shapes[scenario]
        return pytest.param(op_name, M, N, dtype, kernel_cls, marks=mark, id=case_id)

    params = []
    for scenario in strategy_shapes:
        mark = pytest.mark.smoke if scenario == ref_scenario else pytest.mark.full
        params.append(case(sweep_op, scenario, ref_dtype, sweep_cls, mark))
    for dtype in strategy_dtypes[1:]:
        params.append(case(sweep_op, ref_scenario, dtype, sweep_cls, pytest.mark.full))
    for op_name, kernel_cls in sentinels:
        params.append(case(op_name, ref_scenario, ref_dtype, kernel_cls, pytest.mark.full))
    return params
