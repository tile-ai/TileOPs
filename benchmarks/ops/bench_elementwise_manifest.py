"""Benchmarks for every elementwise manifest entry, one test per op over its manifest calls.

Each case is one workload row and dtype case; ``ElementwiseCall`` draws its inputs and holds
the op's reference. Every row is timed against the reference in torch eager and through
inductor. The fused gated ops are benchmarked in ``bench_binary_elementwise.py``, beside
flashinfer's kernels.
"""

import pytest

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    compiled_reference,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.elementwise import (
    AbsFwdOp,
    AddFwdOp,
    AlibiFwdOp,
    BitwiseAndFwdOp,
    BitwiseNotFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
    CeilFwdOp,
    ClampFwdOp,
    ClampScalarFwdOp,
    CosFwdOp,
    DivFwdOp,
    DropoutFwdOp,
    EluFwdOp,
    EqFwdOp,
    ErfFwdOp,
    ExpFwdOp,
    Expm1FwdOp,
    FloorDivideFwdOp,
    FloorFwdOp,
    GeFwdOp,
    GeluFwdOp,
    GtFwdOp,
    HardsigmoidFwdOp,
    HardswishFwdOp,
    HardtanhFwdOp,
    IsfiniteFwdOp,
    IsinfFwdOp,
    IsnanFwdOp,
    LeakyReluFwdOp,
    LeFwdOp,
    LerpFwdOp,
    LerpTensorFwdOp,
    Log1pFwdOp,
    LogFwdOp,
    LogicalAndFwdOp,
    LogicalNotFwdOp,
    LogicalOrFwdOp,
    LtFwdOp,
    MaskedFillFwdOp,
    MaskedFillScalarFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MishFwdOp,
    MulFwdOp,
    NanToNumFwdOp,
    NeFwdOp,
    NegFwdOp,
    PowFwdOp,
    PreluFwdOp,
    ReciprocalFwdOp,
    ReluFwdOp,
    RemainderFwdOp,
    RoundFwdOp,
    RsqrtFwdOp,
    SeluFwdOp,
    SigmoidFwdOp,
    SignFwdOp,
    SiluFwdOp,
    SinFwdOp,
    SinusoidalFwdOp,
    SoftplusFwdOp,
    SqrtFwdOp,
    SubFwdOp,
    TanhFwdOp,
    TruncFwdOp,
    WhereFwdOp,
)
from workloads.elementwise import ElementwiseCall


def _bench(op_cls, call, *, torch_tag: str = "torch", count_copies: bool = False):
    """Time the op against its reference, in torch eager and through inductor."""
    workload = ElementwiseCall(call)
    op = op_cls(**workload.arguments())
    inputs = workload.gen_inputs()
    functors = {
        "tileops": op,
        torch_tag: workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
    }
    ManifestBenchmark(op, workload).compare(functors, *inputs, count_copies=count_copies)


@pytest.mark.parametrize("call", manifest_calls(PreluFwdOp))
def test_prelu_bench(call) -> None:
    _bench(PreluFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(MaskedFillFwdOp))
def test_masked_fill_bench(call) -> None:
    # The baseline is a clone plus an in-place fill, and the clone is a copy, not a
    # kernel; counting copies is what puts all of it in the reading.
    _bench(MaskedFillFwdOp, call, count_copies=True)


@pytest.mark.parametrize("call", manifest_calls(MaskedFillScalarFwdOp))
def test_masked_fill_scalar_bench(call) -> None:
    _bench(MaskedFillScalarFwdOp, call, count_copies=True)


@pytest.mark.parametrize("call", manifest_calls(AddFwdOp))
def test_add_bench(call) -> None:
    _bench(AddFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(SubFwdOp))
def test_sub_bench(call) -> None:
    _bench(SubFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(MulFwdOp))
def test_mul_bench(call) -> None:
    _bench(MulFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(DivFwdOp))
def test_div_bench(call) -> None:
    _bench(DivFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RemainderFwdOp))
def test_remainder_bench(call) -> None:
    _bench(RemainderFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(PowFwdOp))
def test_pow_bench(call) -> None:
    _bench(PowFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(FloorDivideFwdOp))
def test_floor_divide_bench(call) -> None:
    _bench(FloorDivideFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LerpFwdOp))
def test_lerp_bench(call) -> None:
    _bench(LerpFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(MaximumFwdOp))
def test_maximum_bench(call) -> None:
    _bench(MaximumFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(MinimumFwdOp))
def test_minimum_bench(call) -> None:
    _bench(MinimumFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(EqFwdOp))
def test_eq_bench(call) -> None:
    _bench(EqFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(NeFwdOp))
def test_ne_bench(call) -> None:
    _bench(NeFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(GtFwdOp))
def test_gt_bench(call) -> None:
    _bench(GtFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LtFwdOp))
def test_lt_bench(call) -> None:
    _bench(LtFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(GeFwdOp))
def test_ge_bench(call) -> None:
    _bench(GeFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LeFwdOp))
def test_le_bench(call) -> None:
    _bench(LeFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LogicalAndFwdOp))
def test_logical_and_bench(call) -> None:
    _bench(LogicalAndFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LogicalOrFwdOp))
def test_logical_or_bench(call) -> None:
    _bench(LogicalOrFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(BitwiseAndFwdOp))
def test_bitwise_and_bench(call) -> None:
    _bench(BitwiseAndFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(BitwiseOrFwdOp))
def test_bitwise_or_bench(call) -> None:
    _bench(BitwiseOrFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(BitwiseXorFwdOp))
def test_bitwise_xor_bench(call) -> None:
    _bench(BitwiseXorFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(AlibiFwdOp))
def test_alibi_bench(call) -> None:
    _bench(AlibiFwdOp, call, torch_tag="torch-ref")


@pytest.mark.parametrize("call", manifest_calls(SinusoidalFwdOp))
def test_sinusoidal_bench(call) -> None:
    _bench(SinusoidalFwdOp, call, torch_tag="torch-ref")


@pytest.mark.parametrize("call", manifest_calls(WhereFwdOp))
def test_where_bench(call) -> None:
    _bench(WhereFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LerpTensorFwdOp))
def test_lerp_tensor_bench(call) -> None:
    _bench(LerpTensorFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(ReluFwdOp))
def test_relu_bench(call) -> None:
    _bench(ReluFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(GeluFwdOp))
def test_gelu_bench(call) -> None:
    _bench(GeluFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(SiluFwdOp))
def test_silu_bench(call) -> None:
    _bench(SiluFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(HardswishFwdOp))
def test_hardswish_bench(call) -> None:
    _bench(HardswishFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(HardsigmoidFwdOp))
def test_hardsigmoid_bench(call) -> None:
    _bench(HardsigmoidFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(MishFwdOp))
def test_mish_bench(call) -> None:
    _bench(MishFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(SeluFwdOp))
def test_selu_bench(call) -> None:
    _bench(SeluFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LeakyReluFwdOp))
def test_leaky_relu_bench(call) -> None:
    _bench(LeakyReluFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(EluFwdOp))
def test_elu_bench(call) -> None:
    _bench(EluFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(HardtanhFwdOp))
def test_hardtanh_bench(call) -> None:
    _bench(HardtanhFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(SoftplusFwdOp))
def test_softplus_bench(call) -> None:
    _bench(SoftplusFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(ClampFwdOp))
def test_clamp_bench(call) -> None:
    _bench(ClampFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(ClampScalarFwdOp))
def test_clamp_scalar_bench(call) -> None:
    _bench(ClampScalarFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(NanToNumFwdOp))
def test_nan_to_num_bench(call) -> None:
    _bench(NanToNumFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(ExpFwdOp))
def test_exp_bench(call) -> None:
    _bench(ExpFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LogFwdOp))
def test_log_bench(call) -> None:
    _bench(LogFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(SqrtFwdOp))
def test_sqrt_bench(call) -> None:
    _bench(SqrtFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RsqrtFwdOp))
def test_rsqrt_bench(call) -> None:
    _bench(RsqrtFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(AbsFwdOp))
def test_abs_bench(call) -> None:
    _bench(AbsFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(NegFwdOp))
def test_neg_bench(call) -> None:
    _bench(NegFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(ReciprocalFwdOp))
def test_reciprocal_bench(call) -> None:
    _bench(ReciprocalFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(SignFwdOp))
def test_sign_bench(call) -> None:
    _bench(SignFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(SinFwdOp))
def test_sin_bench(call) -> None:
    _bench(SinFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(CosFwdOp))
def test_cos_bench(call) -> None:
    _bench(CosFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(FloorFwdOp))
def test_floor_bench(call) -> None:
    _bench(FloorFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(CeilFwdOp))
def test_ceil_bench(call) -> None:
    _bench(CeilFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RoundFwdOp))
def test_round_bench(call) -> None:
    _bench(RoundFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(TruncFwdOp))
def test_trunc_bench(call) -> None:
    _bench(TruncFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(ErfFwdOp))
def test_erf_bench(call) -> None:
    _bench(ErfFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(Log1pFwdOp))
def test_log1p_bench(call) -> None:
    _bench(Log1pFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(Expm1FwdOp))
def test_expm1_bench(call) -> None:
    _bench(Expm1FwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(SigmoidFwdOp))
def test_sigmoid_bench(call) -> None:
    _bench(SigmoidFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(TanhFwdOp))
def test_tanh_bench(call) -> None:
    _bench(TanhFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LogicalNotFwdOp))
def test_logical_not_bench(call) -> None:
    _bench(LogicalNotFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(BitwiseNotFwdOp))
def test_bitwise_not_bench(call) -> None:
    _bench(BitwiseNotFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(IsnanFwdOp))
def test_isnan_bench(call) -> None:
    _bench(IsnanFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(IsinfFwdOp))
def test_isinf_bench(call) -> None:
    _bench(IsinfFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(IsfiniteFwdOp))
def test_isfinite_bench(call) -> None:
    _bench(IsfiniteFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(DropoutFwdOp))
def test_dropout_bench(call) -> None:
    _bench(DropoutFwdOp, call)
