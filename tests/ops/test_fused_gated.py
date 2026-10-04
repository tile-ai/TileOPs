"""Tests for fused gated elementwise ops (silu_and_mul, gelu_and_mul, gelu_tanh_and_mul).

Covers L1 smoke correctness, multi-dtype coverage, and strategy selection.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.elementwise import (
    FusedGatedKernel,
    GeluAndMulFwdKernel,
    GeluTanhAndMulFwdKernel,
    SiluAndMulFwdKernel,
)
from tileops.ops.elementwise import GeluAndMulFwdOp, GeluTanhAndMulFwdOp, SiluAndMulFwdOp
from tileops.ops.elementwise._base import ELEMENTWISE
from workloads.device import run_device
from workloads.elementwise import (
    GeluAndMulCase,
    GeluTanhAndMulCase,
    SiluAndMulCase,
)


class SiluAndMulFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
                pytest.param(2048, 2048, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class SiluAndMulTest(SiluAndMulCase, TestBase):
    pass


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-2, 1e-2
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@SiluAndMulFixture
def test_silu_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = SiluAndMulTest(m, n, dtype)
    op = SiluAndMulFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_silu_and_mul_lazy_op_rebinds_shape() -> None:
    """Lazy construction should not lock the op to the first runtime shape."""
    op = SiluAndMulFwdOp()
    for m, n in [(32, 64), (16, 128)]:
        test = SiluAndMulTest(m, n, torch.float16)
        test.check(op, *test.gen_inputs())


class GeluAndMulFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class GeluAndMulTest(GeluAndMulCase, TestBase):
    pass


@GeluAndMulFixture
def test_gelu_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluAndMulTest(m, n, dtype)
    op = GeluAndMulFwdOp()
    test.check(op, *test.gen_inputs())


class GeluTanhAndMulFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2048, 2048, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class GeluTanhAndMulTest(GeluTanhAndMulCase, TestBase):
    pass


@GeluTanhAndMulFixture
def test_gelu_tanh_and_mul_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = GeluTanhAndMulTest(m, n, dtype)
    op = GeluTanhAndMulFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_fused_gated_rejects_integer_dtype() -> None:
    """Fused gated ops are float-only; the rejection follows the tensor."""
    op = GeluAndMulFwdOp()
    x = torch.zeros(16, 32, device=run_device(), dtype=torch.int32)
    # The manifest dtype union rejects it before any kernel is asked for.
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x)


@pytest.mark.smoke
def test_fused_gated_serves_two_dtypes_from_one_instance() -> None:
    """The element type comes from the tensor, so both are valid on one op."""
    op = SiluAndMulFwdOp()
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(16, 16, device=run_device(), dtype=dtype)
        assert op(x).dtype == dtype
    assert len(op.built_kernels(ELEMENTWISE)) == 2


@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_silu_and_mul_config_is_in_the_autotune_space(tune: bool) -> None:
    """The shipped config is one the tuner can land on, and tune=True tunes instead of falling back."""
    import warnings

    op = SiluAndMulFwdOp(tune=tune)
    x = torch.randn(1024, 2 * 4096, device=run_device(), dtype=torch.float16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        op(x)
    assert not [w for w in caught if "falling back" in str(w.message)], (
        f"fell back instead of tuning: {[str(w.message) for w in caught]}"
    )
    (kernel,) = op.iter_kernels()
    assert any(
        c["threads"] == kernel.config["threads"]
        and c["num_per_thread"] == kernel.config["num_per_thread"]
        for c in kernel.autotune_configs
    )


# Strategy selection tests


@pytest.mark.smoke
def test_fused_gated_kernel_has_strategies() -> None:
    """FusedGatedKernel must expose STRATEGIES and DEFAULT_STRATEGY class attrs."""
    assert hasattr(FusedGatedKernel, "STRATEGIES")
    assert hasattr(FusedGatedKernel, "DEFAULT_STRATEGY")
    assert "direct" in FusedGatedKernel.STRATEGIES
    assert "explicit_parallel" in FusedGatedKernel.STRATEGIES
    assert FusedGatedKernel.DEFAULT_STRATEGY in FusedGatedKernel.STRATEGIES


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_fused_gated_kernel_rejects_unknown_strategy() -> None:
    """FusedGatedKernel must reject unknown strategy names."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        SiluAndMulFwdKernel(
            M=16,
            N=16,
            dtype=torch.float16,
            config={"strategy": "nonexistent"},
        )


class FusedGatedDirectStrategyFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 1024, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024, 1024, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@pytest.mark.cuda_only
@FusedGatedDirectStrategyFixture
def test_silu_and_mul_direct_strategy(m: int, n: int, dtype: torch.dtype) -> None:
    """SiluAndMul with config strategy='direct' produces correct results."""
    test = SiluAndMulTest(m, n, dtype)
    kernel = SiluAndMulFwdKernel(M=m, N=n, dtype=dtype, config={"strategy": "direct"})
    test.check(SiluAndMulFwdOp(), *test.gen_inputs(), runs=kernel)


@pytest.mark.cuda_only
@FusedGatedDirectStrategyFixture
def test_gelu_and_mul_direct_strategy(m: int, n: int, dtype: torch.dtype) -> None:
    """GeluAndMul with config strategy='direct' produces correct results."""
    test = GeluAndMulTest(m, n, dtype)
    kernel = GeluAndMulFwdKernel(M=m, N=n, dtype=dtype, config={"strategy": "direct"})
    test.check(GeluAndMulFwdOp(), *test.gen_inputs(), runs=kernel)


@pytest.mark.cuda_only
@FusedGatedDirectStrategyFixture
def test_gelu_tanh_and_mul_direct_strategy(m: int, n: int, dtype: torch.dtype) -> None:
    """GeluTanhAndMul with config strategy='direct' produces correct results."""
    test = GeluTanhAndMulTest(m, n, dtype)
    kernel = GeluTanhAndMulFwdKernel(
        M=m,
        N=n,
        dtype=dtype,
        config={"strategy": "direct"},
    )
    test.check(GeluTanhAndMulFwdOp(), *test.gen_inputs(), runs=kernel)


@pytest.mark.smoke
def test_fused_gated_default_strategy_is_explicit_parallel() -> None:
    """Default strategy for FusedGatedKernel should be explicit_parallel."""
    assert FusedGatedKernel.DEFAULT_STRATEGY == "explicit_parallel"


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_fused_gated_kernel_stores_strategy() -> None:
    """FusedGatedKernel records the config-selected strategy on kernel and config."""
    k = SiluAndMulFwdKernel(M=16, N=16, dtype=torch.float16, config={"strategy": "direct"})
    assert k.strategy == "direct"
    assert k.config["strategy"] == "direct"
    k2 = SiluAndMulFwdKernel(M=16, N=16, dtype=torch.float16)
    assert k2.strategy == FusedGatedKernel.DEFAULT_STRATEGY
    assert k2.config["strategy"] == FusedGatedKernel.DEFAULT_STRATEGY
