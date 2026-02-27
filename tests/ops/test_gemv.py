from typing import Tuple

import torch
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops import GemvOp


class GemvFixture(FixtureBase):
    PARAMS = [
        ("n, k, dtype, tune", [
            (1024, 1024, torch.float16, False),
            (7168, 16384, torch.float16, True),
            (18432, 7168, torch.float16, True),
            (1024, 1024, torch.bfloat16, False),
            (7168, 16384, torch.bfloat16, True),
            (18432, 7168, torch.bfloat16, True),
        ]),
    ]


class GemvTest(TestBase):

    def __init__(self, n: int, k: int, dtype: torch.dtype):
        self.n = n
        self.k = k
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        shape_a = (self.k,)
        a = torch.randn(*shape_a, device='cuda', dtype=self.dtype)
        shape_b = (self.n, self.k)
        b = torch.randn(*shape_b, device='cuda', dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return b @ a


@GemvFixture
def test_gemv(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    test = GemvTest(n, k, dtype)
    op = GemvOp(n, k, dtype=dtype, tune=tune)
    if dtype == torch.float16:
        test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-3)
    else:
        test.check(op, *test.gen_inputs(), atol=1.6e-2, rtol=1.6e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
