import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormOp


class FusedAddLayerNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes -- fp32
            pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
            # Standard aligned shapes -- fp16
            pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
            # Standard aligned shapes -- bf16
            pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            # Non-power-of-two hidden dims
            pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
            pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
            # Tail-M: M not divisible by block_m
            pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


class FusedAddLayerNormTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        residual = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, residual, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        add_result = (x.float() + residual.float()).to(x.dtype)
        y = F.layer_norm(
            add_result.float(),
            (self.n,),
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)
        return y, add_result


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@FusedAddLayerNormFixture
def test_fused_add_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddLayerNormTest(m, n, dtype)
    op = FusedAddLayerNormOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class FusedAddLayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 4096, torch.float32, marks=pytest.mark.full),
            pytest.param(1024, 4096, torch.float16, marks=pytest.mark.full),
            pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@FusedAddLayerNormNonContigFixture
def test_fused_add_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    r_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice
    residual = r_full[:, :n]
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")

    op = FusedAddLayerNormOp(M=m, N=n, dtype=dtype)

    # Reference on contiguous copies
    test = FusedAddLayerNormTest(m, n, dtype)
    y_ref, add_ref = test.ref_program(x.contiguous(), residual.contiguous(), weight, bias)

    y, residual_out = op(x, residual, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous y test failed, max err: {(y - y_ref).abs().max()}"
    assert torch.allclose(residual_out, add_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous residual_out test failed, max err: {(residual_out - add_ref).abs().max()}"


class FusedAddLayerNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.full),
            pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.full),
            pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@FusedAddLayerNorm3DFixture
def test_fused_add_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    residual = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")
    bias = torch.randn(hidden, dtype=dtype, device="cuda")

    M = batch * seq
    op = FusedAddLayerNormOp(M=M, N=hidden, dtype=dtype)

    test = FusedAddLayerNormTest(M, hidden, dtype)
    y_ref, add_ref = test.ref_program(x, residual, weight, bias)

    y, residual_out = op(x, residual, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"3D y test failed, max err: {(y - y_ref).abs().max()}"
    assert torch.allclose(residual_out, add_ref, atol=atol, rtol=rtol), \
        f"3D residual_out test failed, max err: {(residual_out - add_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
