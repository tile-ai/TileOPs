import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.fused_add_rmsnorm import FusedAddRMSNormFwdOp
from workloads.fused_add_rmsnorm import FusedAddRmsNormTest as _FusedAddRmsNormTestWorkload


class FusedAddRmsNormTest(_FusedAddRmsNormTestWorkload, TestBase):
    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        add_result = (x.float() + residual.float()).to(x.dtype)
        add_f32 = add_result.float()
        rms = torch.sqrt(add_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = ((add_f32 / rms) * weight.float()).to(x.dtype)
        return y, add_result


class FusedAddRmsNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes -- fp16
            pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
            # Standard aligned shapes -- bf16
            pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            # Non-aligned N
            pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2048, 5120, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2048, 5120, torch.bfloat16, False, marks=pytest.mark.full),
            # Tail-M: M not divisible by block_m
            pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@FusedAddRmsNormFixture
def test_fused_add_rmsnorm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddRmsNormTest(m, n, dtype)
    op = FusedAddRMSNormFwdOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class FusedAddRmsNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@FusedAddRmsNormNonContigFixture
def test_fused_add_rmsnorm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    r_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice
    residual = r_full[:, :n]
    weight = torch.randn(n, dtype=dtype, device="cuda")

    op = FusedAddRMSNormFwdOp(M=m, N=n, dtype=dtype)

    # Reference on contiguous copies
    test = FusedAddRmsNormTest(m, n, dtype)
    y_ref, add_ref = test.ref_program(x.contiguous(), residual.contiguous(), weight)

    y, residual_out = op(x, residual, weight)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous y test failed, max err: {(y - y_ref).abs().max()}"
    assert torch.allclose(residual_out, add_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous residual_out test failed, max err: {(residual_out - add_ref).abs().max()}"


class FusedAddRmsNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@FusedAddRmsNorm3DFixture
def test_fused_add_rmsnorm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    residual = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")

    M = batch * seq
    op = FusedAddRMSNormFwdOp(M=M, N=hidden, dtype=dtype)

    test = FusedAddRmsNormTest(M, hidden, dtype)
    y_ref, add_ref = test.ref_program(x, residual, weight)

    y, residual_out = op(x, residual, weight)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"3D y test failed, max err: {(y - y_ref).abs().max()}"
    assert torch.allclose(residual_out, add_ref, atol=atol, rtol=rtol), \
        f"3D residual_out test failed, max err: {(residual_out - add_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
