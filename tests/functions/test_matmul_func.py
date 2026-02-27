from typing import Tuple, Union

import pytest
import torch

from tests.test_base import TestBase, FixtureBase
from tileops.functions import MatMulFunc, matmul


class MatMulFixture(FixtureBase):
    PARAMS = [
        ("m, n, k, dtype, tune", [
            (1024, 1024, 1024, torch.float16, False),
        ]),
    ]


class MatMulTest(TestBase):

    def __init__(self, m: int, n: int, k: int, dtype: torch.dtype, grad: bool = True):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.grad = grad

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.m, self.k, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        b = torch.randn(self.k, self.n, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        return a, b

    def ref_program(
            self, a: torch.Tensor, b: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        output = torch.matmul(a, b)
        if not self.grad:
            return output
        loss = output.sum()
        loss.backward()
        return output, a.grad, b.grad


@MatMulFixture
def test_matmul(m: int, n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    test = MatMulTest(m, n, k, dtype)
    inputs = test.gen_inputs()

    print("=========Testing matmul function inference=========")
    test.check_fn(matmul, *inputs)

    print("=========Testing matmul function class=========")
    fn = MatMulFunc(m, n, k, dtype, tune)
    test.check_fn(fn, *inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
