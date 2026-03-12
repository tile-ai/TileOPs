"""Tests for bitwise elementwise ops (bitwise_not).

Covers L1 smoke correctness (int32, 1M).
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import BitwiseNotOp


class BitwiseFixture(FixtureBase):
    """Parametrize over shapes / dtypes for bitwise ops."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.int32, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int32, marks=pytest.mark.full),
        ]),
    ]


class BitwiseNotTest(TestBase):
    """Test harness for bitwise_not."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randint(-1000, 1000, (self.n_total,), device="cuda", dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return ~x


@BitwiseFixture
def test_bitwise_not(n_total: int, dtype: torch.dtype) -> None:
    test = BitwiseNotTest(n_total, dtype)
    op = BitwiseNotOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
