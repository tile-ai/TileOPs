import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.layer_norm import LayerNormOp


class LayerNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes -- fp32
            (1024, 4096, torch.float32, False),
            (4096, 4096, torch.float32, False),
            (8192, 8192, torch.float32, False),
            # Standard aligned shapes -- fp16
            (1024, 4096, torch.float16, False),
            (4096, 4096, torch.float16, False),
            (8192, 8192, torch.float16, False),
            # Standard aligned shapes -- bf16
            (1024, 4096, torch.bfloat16, False),
            (4096, 4096, torch.bfloat16, False),
            (8192, 8192, torch.bfloat16, False),
            # Non-power-of-two hidden dims
            (1024, 3000, torch.float32, False),
            (1024, 3000, torch.float16, False),
            (1024, 3000, torch.bfloat16, False),
            (2048, 5120, torch.float32, False),
            (2048, 5120, torch.float16, False),
            (2048, 5120, torch.bfloat16, False),
            # Tail-M: M not divisible by block_m
            (1025, 4096, torch.float16, False),
            (1025, 4096, torch.bfloat16, False),
        ]),
    ]


class LayerNormTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, weight, bias

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # AC-9: reference uses torch.nn.functional.layer_norm
        return F.layer_norm(
            x.float(),
            (self.n,),
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1e-2, 1e-2


@LayerNormFixture
def test_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LayerNormTest(m, n, dtype)
    op = LayerNormOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            (1024, 4096, torch.float32),
            (1024, 4096, torch.float16),
            (1024, 4096, torch.bfloat16),
        ]),
    ]


@LayerNormNonContigFixture
def test_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")

    op = LayerNormOp(M=m, N=n, dtype=dtype)

    # Reference using torch.nn.functional.layer_norm
    x_ref = x.contiguous()
    y_ref = F.layer_norm(
        x_ref.float(), (n,),
        weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


class LayerNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            (2, 512, 4096, torch.float32),
            (2, 512, 4096, torch.float16),
            (2, 512, 4096, torch.bfloat16),
        ]),
    ]


@LayerNorm3DFixture
def test_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")
    bias = torch.randn(hidden, dtype=dtype, device="cuda")

    M = batch * seq
    op = LayerNormOp(M=M, N=hidden, dtype=dtype)

    # Reference using torch.nn.functional.layer_norm
    y_ref = F.layer_norm(
        x.float(), (hidden,),
        weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"3D test failed, max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
