import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from workloads.rms_norm import RMSNormTest as _RMSNormTestWorkload


class RMSNormTest(_RMSNormTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x_f32 / rms) * weight.float()).to(x.dtype)


class RMSNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes (AC required)
            pytest.param(1024, 4096, torch.float16, False, marks=[pytest.mark.smoke, pytest.mark.packaging]),
            pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
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


@RMSNormFixture
def test_rms_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RMSNormTest(m, n, dtype)
    op = RMSNormFwdOp(N=n, dtype=dtype)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    rtol = atol
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class RMSNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@RMSNormNonContigFixture
def test_rms_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device="cuda")

    op = RMSNormFwdOp(N=n, dtype=dtype)

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


class RMSNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@RMSNorm3DFixture
def test_rms_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")

    op = RMSNormFwdOp(N=hidden, dtype=dtype)

    # Reference
    eps = 1e-6
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"3D test failed, max err: {(y - y_ref).abs().max()}"


class RMSNormDimAxis1Fixture(FixtureBase):
    PARAMS = [
        ("batch, hidden, seq, dtype", [
            pytest.param(2, 4096, 512, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 4096, 512, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@RMSNormDimAxis1Fixture
def test_rms_norm_dim_axis1(
    batch: int, hidden: int, seq: int, dtype: torch.dtype
) -> None:
    """Test reducing along dim=1 (not -1) with a 3D tensor.

    Exercises the movedim choreography in RowNormOp._flatten_to_2d /
    _trim_and_unflatten — the new code path enabled by this PR. Without
    this test, the dim != -1 branch could regress while the suite passes.
    """
    x = torch.randn(batch, hidden, seq, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")

    op = RMSNormFwdOp(N=hidden, dtype=dtype, dim=1)

    eps = 1e-6
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=1, keepdim=True) + eps)
    # Apply 1-D weight to dim=1 via reshape for broadcasting.
    weight_b = weight.float().view(1, hidden, 1)
    y_ref = ((x_f32 / rms) * weight_b).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"dim=1 test failed, max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
