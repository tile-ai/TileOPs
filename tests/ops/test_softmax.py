from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase


class SoftmaxFixture(FixtureBase):
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
            # Non-power-of-two hidden
            (1024, 3000, torch.float16),
            (2048, 5120, torch.float16),
            (1024, 3000, torch.bfloat16),
            (2048, 5120, torch.bfloat16),
        ]),
    ]


class SoftmaxMultiDimFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            # 3D input: (B, S, H) -> m=B*S
            (2, 512, 4096, torch.float16),
            (2, 512, 4096, torch.bfloat16),
        ]),
    ]


class SoftmaxNonContiguousFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            # Non-contiguous via slice
            (1024, 4096, torch.float16),
            (1024, 4096, torch.bfloat16),
        ]),
    ]


class SoftmaxTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, device='cuda', dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        x_max = x_f32.max(dim=-1, keepdim=True).values
        exp_x = torch.exp(x_f32 - x_max)
        return (exp_x / exp_x.sum(dim=-1, keepdim=True)).to(x.dtype)


@SoftmaxFixture
def test_softmax(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import SoftmaxOp

    test = SoftmaxTest(m, n, dtype)
    op = SoftmaxOp(m, n, dtype=dtype)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@SoftmaxMultiDimFixture
def test_softmax_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops import SoftmaxOp

    m = batch * seq
    test = SoftmaxTest(m, hidden, dtype)
    op = SoftmaxOp(m, hidden, dtype=dtype)
    # Feed 3D input directly — Op should flatten/unflatten internally
    x = torch.randn(batch, seq, hidden, device='cuda', dtype=dtype)
    ref = test.ref_program(x)
    result = op(x)
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    torch.testing.assert_close(result, ref, **tolerances)


@SoftmaxNonContiguousFixture
def test_softmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops import SoftmaxOp

    test = SoftmaxTest(m, n, dtype)
    op = SoftmaxOp(m, n, dtype=dtype)
    # Generate non-contiguous input by creating a larger tensor and slicing
    x_big = torch.randn(m, n * 2, device='cuda', dtype=dtype)
    x = x_big[:, ::2]  # Non-contiguous via stride
    assert not x.is_contiguous()
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, x, **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
