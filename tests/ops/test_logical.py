"""Tests for logical elementwise ops (logical_not).

Covers L1 smoke correctness (fp16, 1M).
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import LogicalNotOp


class LogicalFixture(FixtureBase):
    """Parametrize over shapes / dtypes for logical ops."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class LogicalNotTest(TestBase):
    """Test harness for logical_not."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        # Mix of zeros and non-zeros for logical_not
        mask = torch.rand(self.n_total, device="cuda") > 0.5
        x[mask] = 0.0
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x == 0, torch.ones_like(x), torch.zeros_like(x))


@LogicalFixture
def test_logical_not(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalNotTest(n_total, dtype)
    op = LogicalNotOp(N_total=n_total, dtype=dtype)
    tol = {"atol": 1e-3, "rtol": 1e-3} if dtype == torch.float16 else {"atol": 1e-5, "rtol": 1e-5}
    test.check(op, *test.gen_inputs(), **tol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
