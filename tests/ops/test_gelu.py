from typing import Tuple

import pytest
import torch
import torch.nn.functional as Fn

from tests.test_base import FixtureBase, TestBase


class GeluTanhFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            (1024, 4096, torch.float16),
            (4096, 4096, torch.float16),
            (8192, 8192, torch.float16),
            (1024, 4096, torch.bfloat16),
            (4096, 4096, torch.bfloat16),
            (8192, 8192, torch.bfloat16),
        ]),
    ]


class GeluErfFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            (1024, 4096, torch.float16),
            (4096, 4096, torch.float16),
            (8192, 8192, torch.float16),
            (1024, 4096, torch.bfloat16),
            (4096, 4096, torch.bfloat16),
            (8192, 8192, torch.bfloat16),
        ]),
    ]


class GeluMultiDimFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype, approximate", [
            (2, 512, 4096, torch.float16, 'tanh'),
            (2, 512, 4096, torch.bfloat16, 'tanh'),
            (2, 512, 4096, torch.float16, 'none'),
            (2, 512, 4096, torch.bfloat16, 'none'),
        ]),
    ]


class GeluNonContiguousFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, approximate", [
            (1024, 4096, torch.float16, 'tanh'),
            (1024, 4096, torch.bfloat16, 'tanh'),
            (1024, 4096, torch.float16, 'none'),
            (1024, 4096, torch.bfloat16, 'none'),
        ]),
    ]


class GeluTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype,
                 approximate: str = 'tanh'):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.approximate = approximate

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, device='cuda', dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        return Fn.gelu(x_f32, approximate=self.approximate).to(x.dtype)


@GeluTanhFixture
def test_gelu_tanh(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import GeluOp

    test = GeluTest(m, n, dtype, approximate='tanh')
    op = GeluOp(m, n, dtype=dtype, approximate='tanh')
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GeluErfFixture
def test_gelu_erf(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import GeluOp

    test = GeluTest(m, n, dtype, approximate='none')
    op = GeluOp(m, n, dtype=dtype, approximate='none')
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GeluMultiDimFixture
def test_gelu_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype,
                 approximate: str) -> None:
    from tileops.ops import GeluOp

    m = batch * seq
    test = GeluTest(m, hidden, dtype, approximate=approximate)
    op = GeluOp(m, hidden, dtype=dtype, approximate=approximate)
    x = torch.randn(batch, seq, hidden, device='cuda', dtype=dtype)
    ref = test.ref_program(x)
    result = op(x)
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    torch.testing.assert_close(result, ref, **tolerances)


@GeluNonContiguousFixture
def test_gelu_non_contiguous(m: int, n: int, dtype: torch.dtype,
                             approximate: str) -> None:
    from tileops.ops import GeluOp

    test = GeluTest(m, n, dtype, approximate=approximate)
    op = GeluOp(m, n, dtype=dtype, approximate=approximate)
    x_big = torch.randn(m, n * 2, device='cuda', dtype=dtype)
    x = x_big[:, ::2]
    assert not x.is_contiguous()
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, x, **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
