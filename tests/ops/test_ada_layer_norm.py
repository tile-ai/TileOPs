import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from workloads.ada_layer_norm import AdaLayerNormTest as _AdaLayerNormTestWorkload


class AdaLayerNormTest(_AdaLayerNormTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # AdaLN: y = scale * LayerNorm(x) + shift
        normed = F.layer_norm(
            x.float(),
            (self.n,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        y = scale.float() * normed + shift.float()
        return y.to(x.dtype)


class AdaLayerNormFixture(FixtureBase):
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


@AdaLayerNormFixture
def test_ada_layer_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp(M=m, N=n, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
