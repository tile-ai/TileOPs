from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import InstanceNormOp


class InstanceNormFixture(FixtureBase):
    PARAMS = [
        ("shape, dtype, eps, affine, non_contiguous, tune", [
            ((2, 64, 32, 32), torch.float16, 1e-5, True, False, False),
            ((2, 64, 32, 32), torch.float16, 1e-5, True, True, False),
            ((2, 64, 17, 31), torch.float16, 1e-5, True, False, False),
            ((2, 64, 32, 32), torch.bfloat16, 1e-5, True, False, False),
            ((4, 32, 128), torch.float16, 1e-5, True, False, False),
            ((4, 32, 128), torch.bfloat16, 1e-5, True, False, False),
        ]),
    ]


class InstanceNormTest(TestBase):

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, eps: float, affine: bool,
                 non_contiguous: bool):
        self.shape = shape
        self.dtype = dtype
        self.eps = eps
        self.affine = affine
        self.non_contiguous = non_contiguous

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x = torch.randn(*self.shape, device='cuda', dtype=self.dtype)
        if self.non_contiguous and x.ndim >= 4:
            x = x.transpose(-1, -2)

        c = self.shape[1]
        weight = torch.randn(c, device='cuda', dtype=self.dtype) if self.affine else None
        bias = torch.randn(c, device='cuda', dtype=self.dtype) if self.affine else None
        return x, weight, bias

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor | None,
                    bias: torch.Tensor | None) -> torch.Tensor:
        return F.instance_norm(
            x,
            running_mean=None,
            running_var=None,
            weight=weight,
            bias=bias,
            use_input_stats=True,
            momentum=0.1,
            eps=self.eps,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@InstanceNormFixture
def test_instance_norm(shape: Tuple[int, ...], dtype: torch.dtype, eps: float, affine: bool,
                       non_contiguous: bool, tune: bool) -> None:
    test = InstanceNormTest(shape, dtype, eps, affine, non_contiguous)
    op = InstanceNormOp(num_channels=shape[1], eps=eps, affine=affine, dtype=dtype, tune=tune)

    if dtype == torch.float16:
        tolerances = {"atol": 1e-3, "rtol": 1e-3}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}

    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
