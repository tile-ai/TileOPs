from typing import Tuple

import pytest
import torch
import torch.nn.functional as Fn

from tests.test_base import FixtureBase, TestBase


class GeluAndMulTanhFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            (1024, 4096, torch.float16),
            (4096, 4096, torch.float16),
            (8192, 8192, torch.float16),
            (1024, 4096, torch.bfloat16),
            (4096, 4096, torch.bfloat16),
            (8192, 8192, torch.bfloat16),
            # Non-aligned hidden (not divisible by 256)
            (1024, 3000, torch.float16),
            (1024, 3000, torch.bfloat16),
            (2048, 5120, torch.float16),
            (2048, 5120, torch.bfloat16),
            # Tail-M: M not divisible by block_m, tests ceildiv boundary
            (1025, 4096, torch.float16),
            (1025, 4096, torch.bfloat16),
        ]),
    ]


class GeluAndMulErfFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            (1024, 4096, torch.float16),
            (4096, 4096, torch.float16),
            (8192, 8192, torch.float16),
            (1024, 4096, torch.bfloat16),
            (4096, 4096, torch.bfloat16),
            (8192, 8192, torch.bfloat16),
            # Non-aligned hidden (not divisible by 256)
            (1024, 3000, torch.float16),
            (1024, 3000, torch.bfloat16),
            (2048, 5120, torch.float16),
            (2048, 5120, torch.bfloat16),
            # Tail-M: M not divisible by block_m, tests ceildiv boundary
            (1025, 4096, torch.float16),
            (1025, 4096, torch.bfloat16),
        ]),
    ]


class GeluAndMulMultiDimFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype, approximate", [
            (2, 512, 4096, torch.float16, 'tanh'),
            (2, 512, 4096, torch.bfloat16, 'tanh'),
            (2, 512, 4096, torch.float16, 'none'),
            (2, 512, 4096, torch.bfloat16, 'none'),
        ]),
    ]


class GeluAndMulNonContiguousFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, approximate", [
            (1024, 4096, torch.float16, 'tanh'),
            (1024, 4096, torch.bfloat16, 'tanh'),
            (1024, 4096, torch.float16, 'none'),
            (1024, 4096, torch.bfloat16, 'none'),
        ]),
    ]


class GeluAndMulTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype,
                 approximate: str = 'tanh'):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.approximate = approximate

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, device='cuda', dtype=self.dtype)
        gate = torch.randn(self.m, self.n, device='cuda', dtype=self.dtype)
        return (x, gate)

    def ref_program(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        gate_f32 = gate.float()
        return (Fn.gelu(x_f32, approximate=self.approximate) * gate_f32).to(x.dtype)


@GeluAndMulTanhFixture
def test_gelu_and_mul_tanh(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import GeluAndMulOp

    test = GeluAndMulTest(m, n, dtype, approximate='tanh')
    op = GeluAndMulOp(m, n, dtype=dtype, approximate='tanh')
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GeluAndMulErfFixture
def test_gelu_and_mul_erf(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import GeluAndMulOp

    test = GeluAndMulTest(m, n, dtype, approximate='none')
    op = GeluAndMulOp(m, n, dtype=dtype, approximate='none')
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GeluAndMulMultiDimFixture
def test_gelu_and_mul_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype,
                         approximate: str) -> None:
    from tileops.ops import GeluAndMulOp

    m = batch * seq
    test = GeluAndMulTest(m, hidden, dtype, approximate=approximate)
    op = GeluAndMulOp(m, hidden, dtype=dtype, approximate=approximate)
    x = torch.randn(batch, seq, hidden, device='cuda', dtype=dtype)
    gate = torch.randn(batch, seq, hidden, device='cuda', dtype=dtype)
    ref = test.ref_program(x, gate)
    result = op(x, gate)
    if result.shape != x.shape:
        raise ValueError(f"Shape mismatch: {result.shape} vs {x.shape}")
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    torch.testing.assert_close(result, ref, **tolerances)


@GeluAndMulNonContiguousFixture
def test_gelu_and_mul_non_contiguous(m: int, n: int, dtype: torch.dtype,
                                     approximate: str) -> None:
    from tileops.ops import GeluAndMulOp

    test = GeluAndMulTest(m, n, dtype, approximate=approximate)
    op = GeluAndMulOp(m, n, dtype=dtype, approximate=approximate)
    x_big = torch.randn(m, n * 2, device='cuda', dtype=dtype)
    x = x_big[:, ::2]
    g_big = torch.randn(m, n * 2, device='cuda', dtype=dtype)
    gate = g_big[:, ::2]
    assert not x.is_contiguous()
    assert not gate.is_contiguous()
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, x, gate, **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
