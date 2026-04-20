import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from workloads.fused_add_layer_norm import (
    FusedAddLayerNormTest as _FusedAddLayerNormTestWorkload,
)


class FusedAddLayerNormTest(_FusedAddLayerNormTestWorkload, TestBase):
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


class FusedAddLayerNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes -- fp32
            pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
        ]),
    ]


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
    op = FusedAddLayerNormFwdOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
