import pytest
import torch

from tests.test_base import FixtureBase, TestBase

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class LayerNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes (AC required)
            (1024, 4096, torch.float16, False),
            (1024, 4096, torch.bfloat16, False),
            (4096, 4096, torch.float16, False),
            (4096, 4096, torch.bfloat16, False),
            (8192, 8192, torch.float16, False),
            (8192, 8192, torch.bfloat16, False),
            # Non-aligned N (AC required)
            (1024, 3000, torch.float16, False),
            (1024, 3000, torch.bfloat16, False),
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
        x_f32 = x.float()
        mean = x_f32.mean(dim=-1, keepdim=True)
        var = x_f32.var(dim=-1, keepdim=True, unbiased=False)
        y = (x_f32 - mean) / torch.sqrt(var + self.eps) * weight.float() + bias.float()
        return y.to(x.dtype)


@LayerNormFixture
def test_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    from tileops.ops.layer_norm import LayerNormOp

    test = LayerNormTest(m, n, dtype)
    op = LayerNormOp(M=m, N=n, dtype=dtype)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    rtol = atol
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class LayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            (1024, 4096, torch.float16),
            (1024, 4096, torch.bfloat16),
        ]),
    ]


@LayerNormNonContigFixture
def test_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    from tileops.ops.layer_norm import LayerNormOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")

    op = LayerNormOp(M=m, N=n, dtype=dtype)

    # Reference on contiguous copy
    eps = 1e-5
    x_ref = x.contiguous()
    x_f32 = x_ref.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = x_f32.var(dim=-1, keepdim=True, unbiased=False)
    y_ref = ((x_f32 - mean) / torch.sqrt(var + eps) * weight.float() + bias.float()).to(dtype)

    y = op(x, weight, bias)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


class LayerNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            (2, 512, 4096, torch.float16),
            (2, 512, 4096, torch.bfloat16),
        ]),
    ]


@LayerNorm3DFixture
def test_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    from tileops.ops.layer_norm import LayerNormOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")
    bias = torch.randn(hidden, dtype=dtype, device="cuda")

    M = batch * seq
    op = LayerNormOp(M=M, N=hidden, dtype=dtype)

    # Reference
    eps = 1e-5
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = x_f32.var(dim=-1, keepdim=True, unbiased=False)
    y_ref = ((x_f32 - mean) / torch.sqrt(var + eps) * weight.float() + bias.float()).to(dtype)

    y = op(x, weight, bias)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"3D test failed, max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
