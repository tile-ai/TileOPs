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
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.timing import bench_kernel, median_busy_ms
from benchmarks.verification import Exact
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
    FusedGatedBenchCase,
)
from workloads.workload_base import FixtureBase

# Scenario -> (tokens, width).
_STRATEGY_SHAPES = {
    "llama-hidden-1k-tokens": (1024, 4096),
    "llama-7b-ffn-1k-tokens": (1024, 11008),
    "llama-hidden-4k-tokens": (4096, 4096),
}
_STRATEGY_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_STRATEGY_KERNELS = [
    ("silu_and_mul", SiluAndMulFwdKernel),
    ("gelu_and_mul", GeluAndMulFwdKernel),
    ("gelu_tanh_and_mul", GeluTanhAndMulFwdKernel),
]
# How far behind the fastest strategy the default may sit before the choice is
# stale. Wide enough to clear run-to-run spread, narrow enough to flag a flip.
_STRATEGY_MARGIN = 1.25


# flashinfer names its fused gated kernels after the same three activations and
# takes the same concatenated input, so a key here doubles as its entry-point name.
def _profile_fused_gated(op_cls, call, library: str) -> None:
    workload = ElementwiseCall(call)
    op = op_cls(**workload.arguments())
    inputs = workload.gen_inputs()
    flashinfer_fn = flashinfer_op(library)
    # Fused activation and multiply: use the same bound as test_fused_gated.py.
    tolerance = (
        {"rtol": 1e-2, "atol": 1e-2}
        if workload.dtype == torch.float16
        else reference_tolerance(workload.dtype)
    )
    ManifestBenchmark(op, workload).compare(
        {
            "tileops": op,
            FLASHINFER_TAG: flashinfer_fn,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
        evidence={
            tag: Exact(**tolerance) for tag in ("tileops", FLASHINFER_TAG, TORCH_COMPILE_TAG)
        },
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
    (sweep_op, sweep_cls), sentinels = _STRATEGY_KERNELS[0], _STRATEGY_KERNELS[1:]
    ref_scenario, ref_dtype = next(iter(_STRATEGY_SHAPES)), torch.float16

    def case(op_name, scenario, dtype, kernel_cls, mark):
        case_id = f"{op_name}-{scenario}-{str(dtype).removeprefix('torch.')}"
        M, N = _STRATEGY_SHAPES[scenario]
        return pytest.param(op_name, M, N, dtype, kernel_cls, marks=mark, id=case_id)

    params = []
    for scenario in _STRATEGY_SHAPES:
        mark = pytest.mark.smoke if scenario == ref_scenario else pytest.mark.full
        params.append(case(sweep_op, scenario, ref_dtype, sweep_cls, mark))
    for dtype in _STRATEGY_DTYPES[1:]:
        params.append(case(sweep_op, ref_scenario, dtype, sweep_cls, pytest.mark.full))
    for op_name, kernel_cls in sentinels:
        params.append(case(op_name, ref_scenario, ref_dtype, kernel_cls, pytest.mark.full))
    return params


class FusedGatedStrategyBenchFixture(FixtureBase):
    PARAMS = [("op_name, M, N, dtype, kernel_cls", _strategy_params())]


@FusedGatedStrategyBenchFixture
def test_fused_gated_default_strategy_is_the_fast_one(
    op_name: str,
    M: int,
    N: int,
    dtype: torch.dtype,
    kernel_cls,
) -> None:
    """The kernel's DEFAULT_STRATEGY is the one that runs fastest here.

    A decision, not a tracked number: it publishes no row, because the report's
    rows are ops and a forced strategy is not one — the Op layer has no way to
    ask for it.
    """
    inputs = FusedGatedBenchCase(M, N, dtype).gen_inputs()

    timings = {}
    for strategy in ("direct", "explicit_parallel"):
        kernel = kernel_cls(M=M, N=N, dtype=dtype, config={"strategy": strategy})
        with torch.no_grad():
            timings[strategy] = median_busy_ms(bench_kernel(kernel, args=inputs))

    default = kernel_cls.DEFAULT_STRATEGY
    fastest = min(timings, key=timings.get)
    assert timings[default] <= timings[fastest] * _STRATEGY_MARGIN, (
        f"{kernel_cls.__name__} {M}x{N} {dtype}: DEFAULT_STRATEGY is {default} at "
        f"{timings[default] * 1e3:.2f}us, but {fastest} runs at "
        f"{timings[fastest] * 1e3:.2f}us"
    )
