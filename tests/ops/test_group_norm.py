import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.group_norm import GroupNormFwdOp
from workloads.group_norm import GroupNormTest as _GroupNormTestWorkload


class GroupNormTest(_GroupNormTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x.float(),
            self.g,
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)


class GroupNormFixture(FixtureBase):
    PARAMS = [
        ("n, c, spatial, g, dtype, tune", [
            # Small CI-friendly shapes -- fp32
            pytest.param(2, 32, (8, 8), 8, torch.float32, False, marks=pytest.mark.smoke),
            # Small CI-friendly shapes -- fp16
            pytest.param(2, 32, (8, 8), 8, torch.float16, False, marks=pytest.mark.smoke),
            # Small CI-friendly shapes -- bf16
            pytest.param(2, 32, (8, 8), 8, torch.bfloat16, False, marks=pytest.mark.smoke),
        ]),
    ]


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@GroupNormFixture
def test_group_norm_op(n: int, c: int, spatial: tuple, g: int,
                       dtype: torch.dtype, tune: bool) -> None:
    test = GroupNormTest(n, c, spatial, g, dtype)
    op = GroupNormFwdOp(N=n, C=c, spatial=spatial, G=g, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
