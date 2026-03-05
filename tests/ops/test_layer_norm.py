from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase


class LayerNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, eps", [
            # Standard shapes - fp16
            (1024, 4096, torch.float16, 1e-5),
            (4096, 4096, torch.float16, 1e-5),
            (8192, 8192, torch.float16, 1e-5),
            # Standard shapes - bf16
            (1024, 4096, torch.bfloat16, 1e-5),
            (4096, 4096, torch.bfloat16, 1e-5),
            (8192, 8192, torch.bfloat16, 1e-5),
            # Non-power-of-two hidden
            (1024, 3000, torch.float16, 1e-5),
            (2048, 5120, torch.float16, 1e-5),
            (1024, 3000, torch.bfloat16, 1e-5),
            (2048, 5120, torch.bfloat16, 1e-5),
        ]),
    ]


class LayerNormMultiDimFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype, eps", [
            # 3D input: (B, S, H) -> m=B*S
            (2, 512, 4096, torch.float16, 1e-5),
            (2, 512, 4096, torch.bfloat16, 1e-5),
        ]),
    ]


class LayerNormNonContiguousFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, eps", [
            # Non-contiguous via slice
            (1024, 4096, torch.float16, 1e-5),
            (1024, 4096, torch.bfloat16, 1e-5),
        ]),
    ]


class LayerNormTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, device='cuda', dtype=self.dtype)
        weight = torch.randn(self.n, device='cuda', dtype=self.dtype)
        bias = torch.randn(self.n, device='cuda', dtype=self.dtype)
        return x, weight, bias

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        mean = x_f32.mean(-1, keepdim=True)
        variance = ((x_f32 - mean) ** 2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.eps)
        return ((x_f32 - mean) * rstd).to(x.dtype) * weight + bias


@LayerNormFixture
def test_layer_norm(m: int, n: int, dtype: torch.dtype, eps: float) -> None:
    from tileops.ops import LayerNormOp

    test = LayerNormTest(m, n, dtype, eps)
    op = LayerNormOp(m, n, eps=eps, dtype=dtype)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@LayerNormMultiDimFixture
def test_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype,
                       eps: float) -> None:
    from tileops.ops import LayerNormOp

    m = batch * seq
    test = LayerNormTest(m, hidden, dtype, eps)
    op = LayerNormOp(m, hidden, eps=eps, dtype=dtype)
    # Feed 3D input directly — Op should flatten/unflatten internally
    x = torch.randn(batch, seq, hidden, device='cuda', dtype=dtype)
    weight = torch.randn(hidden, device='cuda', dtype=dtype)
    bias = torch.randn(hidden, device='cuda', dtype=dtype)
    ref = test.ref_program(x, weight, bias)
    result = op(x, weight, bias)
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    torch.testing.assert_close(result, ref, **tolerances)


@LayerNormNonContiguousFixture
def test_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype,
                                   eps: float) -> None:
    from tileops.ops import LayerNormOp

    test = LayerNormTest(m, n, dtype, eps)
    op = LayerNormOp(m, n, eps=eps, dtype=dtype)
    # Generate non-contiguous input by creating a larger tensor and slicing
    x_big = torch.randn(m, n * 2, device='cuda', dtype=dtype)
    x = x_big[:, ::2]  # Non-contiguous via stride
    assert not x.is_contiguous()
    weight = torch.randn(n, device='cuda', dtype=dtype)
    bias = torch.randn(n, device='cuda', dtype=dtype)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-2, "rtol": 1e-2}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, x, weight, bias, **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
