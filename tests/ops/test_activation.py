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
            # Full: larger follow-up coverage
            pytest.param(4_000_000, torch.float16, marks=pytest.mark.full),
            pytest.param(4_000_000, torch.bfloat16, marks=pytest.mark.full),
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


class ReluStrategyFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype, strategy", [
            pytest.param(1_000_000, torch.float16, "direct", marks=pytest.mark.smoke),
            pytest.param(1_000_000, torch.float16, "explicit_parallel", marks=pytest.mark.full),
            pytest.param(1_000_000, torch.float16, "register_copy", marks=pytest.mark.full),
        ]),
    ]


@ReluStrategyFixture
def test_relu_strategies(n_total: int, dtype: torch.dtype, strategy: str) -> None:
    """Verify all 3 unary strategies produce correct results."""
    test = ReluTest(n_total, dtype)
    op = ReluFwdOp(N_total=n_total, dtype=dtype, strategy=strategy)
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


class ActivationEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
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


@pytest.mark.smoke
def test_activation_rejects_non_float_dtype() -> None:
    from tileops.kernels.elementwise import GeluFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        GeluFwdKernel(N_total=16, dtype=torch.int32)


# ---------------------------------------------------------------------------
# L4 edge-case tests (fp32, 4K)
# ---------------------------------------------------------------------------


@ActivationEdgeFixture
def test_sigmoid_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: sigmoid of large negative -> ~0, large positive -> ~1."""
    from tileops.ops.elementwise import SigmoidFwdOp

    def _extreme(n, dtype):
        x = torch.zeros(n, device="cuda", dtype=dtype)
        x[:n // 2] = -50.0
        x[n // 2:] = 50.0
        return x

    _make_activation_test(n_total, dtype, _extreme, torch.sigmoid, SigmoidFwdOp)


@ActivationEdgeFixture
def test_tanh_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: tanh saturates to +/-1 for large inputs."""
    from tileops.ops.elementwise import TanhFwdOp

    def _extreme(n, dtype):
        x = torch.zeros(n, device="cuda", dtype=dtype)
        x[:n // 2] = -50.0
        x[n // 2:] = 50.0
        return x

    _make_activation_test(n_total, dtype, _extreme, torch.tanh, TanhFwdOp)


# ===========================================================================
# Independent activation ops
# ===========================================================================


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


@pytest.mark.smoke
def test_prelu_batch_dim() -> None:
    """PReLU with a leading batch dimension: shape (2, 4, 8)."""
    from tileops.ops.elementwise import PreluFwdOp

    dtype = torch.float32
    shape = (2, 4, 8)
    C = 4
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda", dtype=dtype)
    ref = F.prelu(x, weight)
    op = PreluFwdOp(shape=shape, dtype=dtype, num_channels=C)
    out = op(x, weight)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    print("All checks passed for PreluFwdOp batch-dim.")


@pytest.mark.smoke
def test_independent_activation_rejects_non_float_dtype() -> None:
    from tileops.kernels.elementwise import LeakyReluFwdKernel
    with pytest.raises(ValueError, match="only supports dtypes"):
        LeakyReluFwdKernel(N_total=16, dtype=torch.int32)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
