"""Tests for bitwise elementwise ops (bitwise_not).

Covers all bool/integer dtypes aligned with torch.bitwise_not.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare
from tileops.ops.elementwise import BitwiseNotOp


class BitwiseFixture(FixtureBase):
    """Parametrize over torch-supported bitwise_not dtypes."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.bool, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.uint8, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int8, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int16, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int32, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int64, marks=pytest.mark.full),
        ]),
    ]


class BitwiseNotTest(TestBase):
    """Test harness for bitwise_not."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device="cuda") > 0.5
        elif self.dtype == torch.uint8:
            x = torch.randint(0, 256, (self.n_total,), device="cuda", dtype=self.dtype)
        else:
            x = torch.randint(-128, 128, (self.n_total,), device="cuda", dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(x)


@BitwiseFixture
def test_bitwise_not(n_total: int, dtype: torch.dtype) -> None:
    test = BitwiseNotTest(n_total, dtype)
    op = BitwiseNotOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


@pytest.mark.parametrize("dtype", [
    pytest.param(torch.float16, marks=pytest.mark.smoke),
    pytest.param(torch.bfloat16, marks=pytest.mark.full),
    pytest.param(torch.float32, marks=pytest.mark.full),
])
def test_bitwise_not_rejects_float_dtype(dtype: torch.dtype) -> None:
    from tileops.kernels.elementwise import BitwiseNotKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseNotKernel(N_total=16, dtype=dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
