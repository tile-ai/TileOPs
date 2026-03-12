"""Tests for logical elementwise ops (logical_not).

Covers float, integer, and bool inputs with torch-style bool output.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare
from tileops.ops.elementwise import LogicalNotOp


class LogicalFixture(FixtureBase):
    """Parametrize over supported dtypes for logical_not."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.bool, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.uint8, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int8, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int16, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int32, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int64, marks=pytest.mark.full),
        ]),
    ]


class LogicalNotTest(TestBase):
    """Test harness for logical_not."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device="cuda") > 0.5
            return (x,)

        if self.dtype == torch.uint8:
            x = torch.randint(0, 8, (self.n_total,), device="cuda", dtype=self.dtype)
        elif self.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            x = torch.randint(-4, 4, (self.n_total,), device="cuda", dtype=self.dtype)
        else:
            x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)

        mask = torch.rand(self.n_total, device="cuda") > 0.5
        x[mask] = 0
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(x)


@LogicalFixture
def test_logical_not(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalNotTest(n_total, dtype)
    op = LogicalNotOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
