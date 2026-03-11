"""Tests for unary activation elementwise ops (relu).

Covers L1 smoke correctness and multi-dtype coverage.
"""

import pytest
import torch

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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
