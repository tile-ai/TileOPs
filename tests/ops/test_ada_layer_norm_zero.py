import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from workloads.ada_layer_norm_zero import AdaLayerNormZeroTest as _AdaLayerNormZeroTestWorkload


class AdaLayerNormZeroTest(_AdaLayerNormZeroTestWorkload, TestBase):
    def ref_program(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, gate: torch.Tensor,
    ) -> torch.Tensor:
        # AdaLN-Zero: y = gate * (scale * LayerNorm(x) + shift)
        normed = F.layer_norm(
            x.float(),
            (self.n,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        y = gate.float() * (scale.float() * normed + shift.float())
        return y.to(x.dtype)


class AdaLayerNormZeroFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            # Standard aligned shapes -- fp32
            pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
            # Standard aligned shapes -- fp16
            pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
            # Standard aligned shapes -- bf16
            pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@AdaLayerNormZeroFixture
def test_ada_layer_norm_zero_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    op = AdaLayerNormZeroFwdOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
