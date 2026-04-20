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
        ]),
    ]


@RMSNormFixture
def test_rms_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RMSNormTest(m, n, dtype)
    op = RMSNormFwdOp(M=m, N=n, dtype=dtype)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    rtol = atol
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
