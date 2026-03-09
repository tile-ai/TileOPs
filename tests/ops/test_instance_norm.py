"""Tests for InstanceNormOp.

InstanceNorm is GroupNorm with G=C. Correctness is validated against
torch.nn.functional.instance_norm.

Run:
    PYTHONPATH="$PWD" python -m pytest tests/ops/test_instance_norm.py -vvs
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.instance_norm import InstanceNormOp


class InstanceNormFixture(FixtureBase):
    PARAMS = [
        ("N, C, spatial, dtype", [
            # 2D spatial -- fp32
            (2, 32, (8, 8), torch.float32),
            (4, 64, (16, 16), torch.float32),
            # 2D spatial -- fp16
            (2, 32, (8, 8), torch.float16),
            (4, 64, (16, 16), torch.float16),
            # 2D spatial -- bf16
            (2, 32, (8, 8), torch.bfloat16),
            (4, 64, (16, 16), torch.bfloat16),
            # 1D spatial
            (2, 32, (64,), torch.float32),
            (2, 32, (64,), torch.float16),
            # Larger spatial
            (2, 16, (32, 32), torch.float16),
        ]),
    ]


class InstanceNormTest(TestBase):

    def __init__(self, N: int, C: int, spatial: tuple, dtype: torch.dtype, eps: float = 1e-5):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (self.N, self.C, *self.spatial)
        x = torch.randn(shape, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.C, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.C, dtype=self.dtype, device="cuda")
        return x, weight, bias

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.instance_norm(
            x.float(),
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@InstanceNormFixture
def test_instance_norm_op(N: int, C: int, spatial: tuple, dtype: torch.dtype) -> None:
    test = InstanceNormTest(N, C, spatial, dtype)
    op = InstanceNormOp(N=N, C=C, spatial=spatial, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
