"""Tests for unary activation elementwise ops.

Covers L1 smoke correctness, multi-dtype coverage, and L4 edge cases.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import ReluFwdOp
from workloads.activation import ReluTest as _ReluTestWorkload


class ReluTest(_ReluTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.float()).to(x.dtype)


class ReluFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            # Smoke: one typical shape per supported dtype
            pytest.param(1_000_000, torch.float16, marks=[pytest.mark.smoke, pytest.mark.packaging]),
            pytest.param(1_000_000, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1_000_000, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@ReluFixture
def test_relu_op(n_total: int, dtype: torch.dtype) -> None:
    test = ReluTest(n_total, dtype)
    op = ReluFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ===========================================================================
# Template-based activation ops
# ===========================================================================


class ActivationFixture(FixtureBase):
    """Parametrize over shapes / dtypes for activation ops."""
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


class UnaryActivationTest(TestBase):
    """Generic test harness for a single-input, single-output unary op."""

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None, ref_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn
        self._ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        return (torch.randn(self.n_total, device="cuda", dtype=self.dtype),)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return self._ref_fn(x)


def _randn(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(n, device="cuda", dtype=dtype)


def _make_activation_test(n_total, dtype, gen_fn, ref_fn, op_cls):
    """Build test, instantiate op, and run check."""
    test = UnaryActivationTest(n_total, dtype, gen_fn=gen_fn, ref_fn=ref_fn)
    op = op_cls(N_total=n_total, dtype=dtype)
    if dtype == torch.float16:
        tol = {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.bfloat16:
        tol = {"atol": 1.6e-2, "rtol": 1.6e-2}
    else:
        tol = {"atol": 1e-5, "rtol": 1e-5}
    test.check(op, *test.gen_inputs(), **tol)


@ActivationFixture
def test_gelu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import GeluFwdOp
    _make_activation_test(n_total, dtype, _randn,
                          F.gelu, GeluFwdOp)


@ActivationFixture
def test_silu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SiluFwdOp
    _make_activation_test(n_total, dtype, _randn, F.silu, SiluFwdOp)


@ActivationFixture
def test_sigmoid(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SigmoidFwdOp
    _make_activation_test(n_total, dtype, _randn, torch.sigmoid, SigmoidFwdOp)


@ActivationFixture
def test_tanh(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import TanhFwdOp
    _make_activation_test(n_total, dtype, _randn, torch.tanh, TanhFwdOp)


@ActivationFixture
def test_hardswish(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import HardswishFwdOp
    _make_activation_test(n_total, dtype, _randn, F.hardswish, HardswishFwdOp)


@ActivationFixture
def test_hardsigmoid(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import HardsigmoidFwdOp
    _make_activation_test(n_total, dtype, _randn, F.hardsigmoid, HardsigmoidFwdOp)


@ActivationFixture
def test_mish(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import MishFwdOp
    _make_activation_test(n_total, dtype, _randn, F.mish, MishFwdOp)


@ActivationFixture
def test_selu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SeluFwdOp
    _make_activation_test(n_total, dtype, _randn, F.selu, SeluFwdOp)


@ActivationFixture
def test_leaky_relu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import LeakyReluFwdOp
    _make_activation_test(
        n_total, dtype, _randn,
        lambda x: F.leaky_relu(x.float(), 0.01).to(x.dtype),
        LeakyReluFwdOp,
    )


@ActivationFixture
def test_elu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import EluFwdOp
    _make_activation_test(
        n_total, dtype, _randn,
        lambda x: F.elu(x.float(), 1.0).to(x.dtype),
        EluFwdOp,
    )


@ActivationFixture
def test_hardtanh(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import HardtanhFwdOp
    _make_activation_test(
        n_total, dtype, _randn,
        lambda x: F.hardtanh(x.float(), -1.0, 1.0).to(x.dtype),
        HardtanhFwdOp,
    )


@ActivationFixture
def test_softplus(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SoftplusFwdOp
    _make_activation_test(
        n_total, dtype, _randn,
        lambda x: F.softplus(x.float(), 1.0, 20.0).to(x.dtype),
        SoftplusFwdOp,
    )


class PreluFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@PreluFixture
def test_prelu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import PreluFwdOp

    C = 64
    H = n_total // C
    # Shape (1, C, H): batch=1, channels=C, spatial=H
    shape = (1, C, H)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(C, device="cuda", dtype=dtype).abs() * 0.1 + 0.01
    ref = F.prelu(x.float(), weight.float()).to(dtype)

    op = PreluFwdOp(shape=shape, dtype=dtype, num_channels=C)
    out = op(x, weight)
    if dtype == torch.float16:
        tol = {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.bfloat16:
        tol = {"atol": 1.6e-2, "rtol": 1.6e-2}
    else:
        tol = {"atol": 1e-5, "rtol": 1e-5}
    torch.testing.assert_close(out, ref, **tol)
    print("All checks passed for PreluFwdOp.")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
