import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from workloads.fused_add_rms_norm import FusedAddRMSNormTest as _FusedAddRMSNormTestWorkload


class FusedAddRMSNormTest(_FusedAddRMSNormTestWorkload, TestBase):
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


class FusedAddRMSNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes -- fp16
            pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
        ]),
    ]


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@FusedAddRMSNormFixture
def test_fused_add_rms_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddRMSNormTest(m, n, dtype)
    op = FusedAddRMSNormFwdOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
