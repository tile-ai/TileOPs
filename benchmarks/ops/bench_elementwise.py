"""Benchmarks for every elementwise manifest entry, one test per op over its manifest calls.

Every row is timed against the reference in torch eager and through inductor. The fused gated
ops are benchmarked in ``bench_fused_gated.py``, beside flashinfer's kernels.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    compiled_reference,
)
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
    CosFwdOp,
    DivFwdOp,
    DropoutFwdOp,
    ELUFwdOp,
    EqFwdOp,
    ErfFwdOp,
    ExpFwdOp,
    Expm1FwdOp,
    FloorDivideFwdOp,
    FloorFwdOp,
    GeFwdOp,
    GELUFwdOp,
    GtFwdOp,
    HardsigmoidFwdOp,
    HardswishFwdOp,
    HardtanhFwdOp,
    IsfiniteFwdOp,
    IsinfFwdOp,
    IsnanFwdOp,
    LeakyReLUFwdOp,
    LeFwdOp,
    LerpFwdOp,
    Log1pFwdOp,
    LogFwdOp,
    LogicalAndFwdOp,
    LogicalNotFwdOp,
    LogicalOrFwdOp,
    LtFwdOp,
    MaskedFillFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MishFwdOp,
    MulFwdOp,
    NanToNumFwdOp,
    NeFwdOp,
    NegFwdOp,
    PowFwdOp,
    PReLUFwdOp,
    ReciprocalFwdOp,
    ReLUFwdOp,
    RemainderFwdOp,
    RoundFwdOp,
    RsqrtFwdOp,
    SELUFwdOp,
    SigmoidFwdOp,
    SignFwdOp,
    SiLUFwdOp,
    SinFwdOp,
    SinusoidalFwdOp,
    SoftplusFwdOp,
    SqrtFwdOp,
    SubFwdOp,
    TanhFwdOp,
    TruncFwdOp,
    WhereFwdOp,
)


def _bench(op_cls, case: bench.Case, *, torch_tag: str = "torch"):
    """Time the op against torch, in eager and through inductor."""
    op = op_cls(**case.arguments)
    bench.Runner(op, case).compare(
        {
            torch_tag: case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )


@pytest.mark.parametrize("case", bench.cases(PReLUFwdOp))
def test_prelu_bench(case) -> None:
    _bench(PReLUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MaskedFillFwdOp))
def test_masked_fill_bench(case) -> None:
    _bench(MaskedFillFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(AddFwdOp))
def test_add_bench(case) -> None:
    _bench(AddFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SubFwdOp))
def test_sub_bench(case) -> None:
    _bench(SubFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MulFwdOp))
def test_mul_bench(case) -> None:
    _bench(MulFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(DivFwdOp))
def test_div_bench(case) -> None:
    _bench(DivFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RemainderFwdOp))
def test_remainder_bench(case) -> None:
    _bench(RemainderFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(PowFwdOp))
def test_pow_bench(case) -> None:
    _bench(PowFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(FloorDivideFwdOp))
def test_floor_divide_bench(case) -> None:
    _bench(FloorDivideFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LerpFwdOp))
def test_lerp_bench(case) -> None:
    _bench(LerpFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MaximumFwdOp))
def test_maximum_bench(case) -> None:
    _bench(MaximumFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MinimumFwdOp))
def test_minimum_bench(case) -> None:
    _bench(MinimumFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(EqFwdOp))
def test_eq_bench(case) -> None:
    _bench(EqFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(NeFwdOp))
def test_ne_bench(case) -> None:
    _bench(NeFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(GtFwdOp))
def test_gt_bench(case) -> None:
    _bench(GtFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LtFwdOp))
def test_lt_bench(case) -> None:
    _bench(LtFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(GeFwdOp))
def test_ge_bench(case) -> None:
    _bench(GeFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LeFwdOp))
def test_le_bench(case) -> None:
    _bench(LeFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LogicalAndFwdOp))
def test_logical_and_bench(case) -> None:
    _bench(LogicalAndFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LogicalOrFwdOp))
def test_logical_or_bench(case) -> None:
    _bench(LogicalOrFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(BitwiseAndFwdOp))
def test_bitwise_and_bench(case) -> None:
    _bench(BitwiseAndFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(BitwiseOrFwdOp))
def test_bitwise_or_bench(case) -> None:
    _bench(BitwiseOrFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(BitwiseXorFwdOp))
def test_bitwise_xor_bench(case) -> None:
    _bench(BitwiseXorFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(AlibiFwdOp))
def test_alibi_bench(case) -> None:
    _bench(AlibiFwdOp, case, torch_tag="torch-ref")


@pytest.mark.parametrize("case", bench.cases(SinusoidalFwdOp))
def test_sinusoidal_bench(case) -> None:
    _bench(SinusoidalFwdOp, case, torch_tag="torch-ref")


@pytest.mark.parametrize("case", bench.cases(WhereFwdOp))
def test_where_bench(case) -> None:
    _bench(WhereFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ReLUFwdOp))
def test_relu_bench(case) -> None:
    _bench(ReLUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(GELUFwdOp))
def test_gelu_bench(case) -> None:
    _bench(GELUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SiLUFwdOp))
def test_silu_bench(case) -> None:
    _bench(SiLUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(HardswishFwdOp))
def test_hardswish_bench(case) -> None:
    _bench(HardswishFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(HardsigmoidFwdOp))
def test_hardsigmoid_bench(case) -> None:
    _bench(HardsigmoidFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(MishFwdOp))
def test_mish_bench(case) -> None:
    _bench(MishFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SELUFwdOp))
def test_selu_bench(case) -> None:
    _bench(SELUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LeakyReLUFwdOp))
def test_leaky_relu_bench(case) -> None:
    _bench(LeakyReLUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ELUFwdOp))
def test_elu_bench(case) -> None:
    _bench(ELUFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(HardtanhFwdOp))
def test_hardtanh_bench(case) -> None:
    _bench(HardtanhFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SoftplusFwdOp))
def test_softplus_bench(case) -> None:
    _bench(SoftplusFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ClampFwdOp))
def test_clamp_bench(case) -> None:
    _bench(ClampFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(NanToNumFwdOp))
def test_nan_to_num_bench(case) -> None:
    _bench(NanToNumFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ExpFwdOp))
def test_exp_bench(case) -> None:
    _bench(ExpFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LogFwdOp))
def test_log_bench(case) -> None:
    _bench(LogFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SqrtFwdOp))
def test_sqrt_bench(case) -> None:
    _bench(SqrtFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RsqrtFwdOp))
def test_rsqrt_bench(case) -> None:
    _bench(RsqrtFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(AbsFwdOp))
def test_abs_bench(case) -> None:
    _bench(AbsFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(NegFwdOp))
def test_neg_bench(case) -> None:
    _bench(NegFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ReciprocalFwdOp))
def test_reciprocal_bench(case) -> None:
    _bench(ReciprocalFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SignFwdOp))
def test_sign_bench(case) -> None:
    _bench(SignFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SinFwdOp))
def test_sin_bench(case) -> None:
    _bench(SinFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(CosFwdOp))
def test_cos_bench(case) -> None:
    _bench(CosFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(FloorFwdOp))
def test_floor_bench(case) -> None:
    _bench(FloorFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(CeilFwdOp))
def test_ceil_bench(case) -> None:
    _bench(CeilFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RoundFwdOp))
def test_round_bench(case) -> None:
    _bench(RoundFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(TruncFwdOp))
def test_trunc_bench(case) -> None:
    _bench(TruncFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ErfFwdOp))
def test_erf_bench(case) -> None:
    _bench(ErfFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(Log1pFwdOp))
def test_log1p_bench(case) -> None:
    _bench(Log1pFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(Expm1FwdOp))
def test_expm1_bench(case) -> None:
    _bench(Expm1FwdOp, case)


@pytest.mark.parametrize("case", bench.cases(SigmoidFwdOp))
def test_sigmoid_bench(case) -> None:
    _bench(SigmoidFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(TanhFwdOp))
def test_tanh_bench(case) -> None:
    _bench(TanhFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LogicalNotFwdOp))
def test_logical_not_bench(case) -> None:
    _bench(LogicalNotFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(BitwiseNotFwdOp))
def test_bitwise_not_bench(case) -> None:
    _bench(BitwiseNotFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(IsnanFwdOp))
def test_isnan_bench(case) -> None:
    _bench(IsnanFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(IsinfFwdOp))
def test_isinf_bench(case) -> None:
    _bench(IsinfFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(IsfiniteFwdOp))
def test_isfinite_bench(case) -> None:
    _bench(IsfiniteFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(DropoutFwdOp))
def test_dropout_bench(case) -> None:
    _bench(DropoutFwdOp, case)
