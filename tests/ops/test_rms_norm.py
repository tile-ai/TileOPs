import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from workloads.ops.rms_norm import RmsNormTest as _RmsNormTestWorkload


class RmsNormTest(_RmsNormTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x_f32 / rms) * weight.float()).to(x.dtype)


class RmsNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes (AC required)
            pytest.param(1024, 4096, torch.float16, False, marks=[pytest.mark.smoke, pytest.mark.packaging]),
            pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(8192, 8192, torch.float16, False, marks=pytest.mark.full),
            pytest.param(8192, 8192, torch.bfloat16, False, marks=pytest.mark.full),
            # Non-aligned N (AC required)
            pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2048, 5120, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2048, 5120, torch.bfloat16, False, marks=pytest.mark.full),
            # Tail-M: M not divisible by block_m (proves T.copy partial block safety)
            pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@RmsNormFixture
def test_rms_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RmsNormTest(m, n, dtype)
    op = RMSNormFwdOp(M=m, N=n, dtype=dtype)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    rtol = atol
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class RmsNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@RmsNormNonContigFixture
def test_rms_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device="cuda")

    op = RMSNormFwdOp(M=m, N=n, dtype=dtype)

    # Reference on contiguous copy
    eps = 1e-6
    x_ref = x.contiguous()
    x_f32 = x_ref.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


class RmsNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@RmsNorm3DFixture
def test_rms_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")

    M = batch * seq
    op = RMSNormFwdOp(M=M, N=hidden, dtype=dtype)

    # Reference
    eps = 1e-6
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"3D test failed, max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
