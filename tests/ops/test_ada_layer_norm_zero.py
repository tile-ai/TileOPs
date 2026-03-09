import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroOp


class AdaLayerNormZeroFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes -- fp32
            (1024, 4096, torch.float32, False),
            (4096, 4096, torch.float32, False),
            # Standard aligned shapes -- fp16
            (1024, 4096, torch.float16, False),
            (4096, 4096, torch.float16, False),
            # Standard aligned shapes -- bf16
            (1024, 4096, torch.bfloat16, False),
            (4096, 4096, torch.bfloat16, False),
            # Non-power-of-two hidden dims
            (1024, 3000, torch.float32, False),
            (1024, 3000, torch.float16, False),
            (1024, 3000, torch.bfloat16, False),
            # Tail-M: M not divisible by block_m
            (1025, 4096, torch.float16, False),
            (1025, 4096, torch.bfloat16, False),
        ]),
    ]


class AdaLayerNormZeroTest(TestBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        scale = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        shift = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        gate = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return x, scale, shift, gate

    def ref_program(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: gate * (scale * LayerNorm(x) + shift) using PyTorch."""
        normed = F.layer_norm(
            x.float(),
            (self.n,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        y = gate.float() * (scale.float() * normed + shift.float())
        return y.to(x.dtype)


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@AdaLayerNormZeroFixture
def test_ada_layer_norm_zero_op(
    m: int, n: int, dtype: torch.dtype, tune: bool,
) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    op = AdaLayerNormZeroOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class AdaLayerNormZero3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            (2, 512, 4096, torch.float32),
            (2, 512, 4096, torch.float16),
            (2, 512, 4096, torch.bfloat16),
        ]),
    ]


@AdaLayerNormZero3DFixture
def test_ada_layer_norm_zero_3d(
    batch: int, seq: int, hidden: int, dtype: torch.dtype,
) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    scale = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    shift = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    gate = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")

    M = batch * seq
    op = AdaLayerNormZeroOp(M=M, N=hidden, dtype=dtype)

    # Reference
    normed = F.layer_norm(
        x.float(), (hidden,), weight=None, bias=None, eps=1e-5,
    )
    y_ref = (gate.float() * (scale.float() * normed + shift.float())).to(dtype)

    y = op(x, scale, shift, gate)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"3D test failed, max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
