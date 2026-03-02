from typing import Tuple

import torch
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops import GemmOp


class GemmFixture(FixtureBase):
    PARAMS = [
        ("m, n, k, dtype, trans_a, trans_b, tune", [
            (1024, 1024, 1024, torch.float16, False, False, False),
            (1, 1024, 1024, torch.float16, False, True, False),
            (1, 7168, 16384, torch.float16, False, True, True),
            (1, 18432, 7168, torch.float16, False, True, True),
            (1024, 1, 1024, torch.float16, False, False, False),
            (7168, 1, 16384, torch.float16, False, False, True),
            (18432, 1, 7168, torch.float16, False, False, True),
            (1, 1024, 1024, torch.bfloat16, False, True, False),
            (1, 7168, 16384, torch.bfloat16, False, True, True),
            (1, 18432, 7168, torch.bfloat16, False, True, True),
            (1024, 1, 1024, torch.bfloat16, False, False, False),
            (7168, 1, 16384, torch.bfloat16, False, False, True),
            (18432, 1, 7168, torch.bfloat16, False, False, True),
        ]),
    ]


class GemmTest(TestBase):

    def __init__(self, m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool = False,
                 trans_b: bool = False):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        shape_a = (self.k, self.m) if self.trans_a else (self.m, self.k)
        a = torch.randn(*shape_a, device='cuda', dtype=self.dtype)
        shape_b = (self.n, self.k) if self.trans_b else (self.k, self.n)
        b = torch.randn(*shape_b, device='cuda', dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


@GemmFixture
def test_gemm(m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool,
              tune: bool) -> None:
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    op = GemmOp(m, n, k, trans_a=trans_a, trans_b=trans_b, dtype=dtype, tune=tune)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-3, "rtol": 1e-3}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
