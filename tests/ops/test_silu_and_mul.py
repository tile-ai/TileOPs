from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase


class SiluAndMulFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            # Standard shapes - fp16
            (1024, 4096, torch.float16),
            (4096, 4096, torch.float16),
            (8192, 8192, torch.float16),
            # Standard shapes - bf16
            (1024, 4096, torch.bfloat16),
            (4096, 4096, torch.bfloat16),
            (8192, 8192, torch.bfloat16),
            # Non-aligned hidden (not divisible by 256) - exercises padding path
            (1024, 3000, torch.float16),
            (1024, 3000, torch.bfloat16),
            (2048, 5120, torch.float16),
            (2048, 5120, torch.bfloat16),
        ]),
    ]


class SiluAndMulMultiDimFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            (2, 512, 4096, torch.float16),
            (2, 512, 4096, torch.bfloat16),
        ]),
    ]


class SiluAndMulNonContiguousFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            (1024, 4096, torch.float16),
            (1024, 4096, torch.bfloat16),
        ]),
    ]


class SiluAndMulTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, device='cuda', dtype=self.dtype)
        gate = torch.randn(self.m, self.n, device='cuda', dtype=self.dtype)
        return (x, gate)

    def ref_program(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        gate_f32 = gate.float()
        return ((x_f32 * torch.sigmoid(x_f32)) * gate_f32).to(x.dtype)


@SiluAndMulFixture
def test_silu_and_mul(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import SiluAndMulOp

    test = SiluAndMulTest(m, n, dtype)
    op = SiluAndMulOp(m, n, dtype=dtype)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@SiluAndMulMultiDimFixture
def test_silu_and_mul_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops import SiluAndMulOp

    m = batch * seq
    test = SiluAndMulTest(m, hidden, dtype)
    op = SiluAndMulOp(m, hidden, dtype=dtype)
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


@SiluAndMulNonContiguousFixture
def test_silu_and_mul_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import SiluAndMulOp

    test = SiluAndMulTest(m, n, dtype)
    op = SiluAndMulOp(m, n, dtype=dtype)
    # Make non-contiguous x
    x_big = torch.randn(m, n * 2, device='cuda', dtype=dtype)
    x = x_big[:, ::2]
    # Make non-contiguous gate
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
