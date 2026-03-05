from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase


class SiluFixture(FixtureBase):
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
        ]),
    ]


class SiluMultiDimFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            # 3D input: (B, S, H) -> m=B*S
            (2, 512, 4096, torch.float16),
            (2, 512, 4096, torch.bfloat16),
        ]),
    ]


class SiluNonContiguousFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            # Non-contiguous via slice
            (1024, 4096, torch.float16),
            (1024, 4096, torch.bfloat16),
        ]),
    ]


class SiluTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, device='cuda', dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        return (x_f32 * torch.sigmoid(x_f32)).to(x.dtype)


@SiluFixture
def test_silu(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import SiluOp

    test = SiluTest(m, n, dtype)
    op = SiluOp(m, n, dtype=dtype)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@SiluMultiDimFixture
def test_silu_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops import SiluOp

    m = batch * seq
    test = SiluTest(m, hidden, dtype)
    op = SiluOp(m, hidden, dtype=dtype)
    x = torch.randn(batch, seq, hidden, device='cuda', dtype=dtype)
    ref = test.ref_program(x)
    result = op(x)
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    torch.testing.assert_close(result, ref, **tolerances)


@SiluNonContiguousFixture
def test_silu_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import SiluOp

    test = SiluTest(m, n, dtype)
    op = SiluOp(m, n, dtype=dtype)
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
