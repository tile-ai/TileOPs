"""Tests for unary activation elementwise ops.

Covers L1 smoke correctness, multi-dtype coverage, and L4 edge cases.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import ReluOp


class ReluFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            # Smoke: fp16, 1M elements
            pytest.param(1_000_000, torch.float16, marks=pytest.mark.smoke),
            # Full: other dtypes and sizes
            pytest.param(1_000_000, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(1_000_000, torch.float32, marks=pytest.mark.full),
            pytest.param(4_000_000, torch.float16, marks=pytest.mark.full),
            pytest.param(4_000_000, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


class ReluTest(TestBase):

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.float()).to(x.dtype)


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
    op = ReluOp(N_total=n_total, dtype=dtype)
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
    op = ReluOp(N_total=n_total, dtype=dtype, strategy=strategy)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ===========================================================================
# 8 activation ops (issue #437)
# ===========================================================================


class ActivationFixture(FixtureBase):
    """Parametrize over shapes / dtypes for activation ops."""
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.full),
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
    tol = {"atol": 1e-3, "rtol": 1e-3} if dtype == torch.float16 else {"atol": 1e-5, "rtol": 1e-5}
    test.check(op, *test.gen_inputs(), **tol)


@ActivationFixture
def test_gelu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import GeluOp
    _make_activation_test(n_total, dtype, _randn,
                          lambda x: F.gelu(x, approximate="tanh"), GeluOp)


@ActivationFixture
def test_silu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SiluOp
    _make_activation_test(n_total, dtype, _randn, F.silu, SiluOp)


@ActivationFixture
def test_sigmoid(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SigmoidOp
    _make_activation_test(n_total, dtype, _randn, torch.sigmoid, SigmoidOp)


@ActivationFixture
def test_tanh(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import TanhOp
    _make_activation_test(n_total, dtype, _randn, torch.tanh, TanhOp)


@ActivationFixture
def test_hardswish(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import HardswishOp
    _make_activation_test(n_total, dtype, _randn, F.hardswish, HardswishOp)


@ActivationFixture
def test_hardsigmoid(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import HardsigmoidOp
    _make_activation_test(n_total, dtype, _randn, F.hardsigmoid, HardsigmoidOp)


@ActivationFixture
def test_mish(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import MishOp
    _make_activation_test(n_total, dtype, _randn, F.mish, MishOp)


@ActivationFixture
def test_selu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SeluOp
    _make_activation_test(n_total, dtype, _randn, F.selu, SeluOp)


# ---------------------------------------------------------------------------
# L4 edge-case tests (fp32, 4K)
# ---------------------------------------------------------------------------


@ActivationEdgeFixture
def test_sigmoid_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: sigmoid of large negative -> ~0, large positive -> ~1."""
    from tileops.ops.elementwise import SigmoidOp

    def _extreme(n, dtype):
        x = torch.zeros(n, device="cuda", dtype=dtype)
        x[:n // 2] = -50.0
        x[n // 2:] = 50.0
        return x

    _make_activation_test(n_total, dtype, _extreme, torch.sigmoid, SigmoidOp)


@ActivationEdgeFixture
def test_tanh_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: tanh saturates to +/-1 for large inputs."""
    from tileops.ops.elementwise import TanhOp

    def _extreme(n, dtype):
        x = torch.zeros(n, device="cuda", dtype=dtype)
        x[:n // 2] = -50.0
        x[n // 2:] = 50.0
        return x

    _make_activation_test(n_total, dtype, _extreme, torch.tanh, TanhOp)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
