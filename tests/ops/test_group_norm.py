"""Tests for GroupNormOp.

Correctness is validated against torch.nn.functional.group_norm.

Run:
    PYTHONPATH="$PWD" python -m pytest tests/ops/test_group_norm.py -vvs
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.group_norm import GroupNormOp


class GroupNormFixture(FixtureBase):
    PARAMS = [
        ("N, C, spatial, G, dtype", [
            # 1D spatial -- fp32
            (2, 64, (32,), 32, torch.float32),
            (4, 128, (64,), 32, torch.float32),
            # 2D spatial -- fp32
            (2, 64, (8, 8), 32, torch.float32),
            (4, 128, (16, 16), 32, torch.float32),
            # 2D spatial -- fp16
            (2, 64, (8, 8), 32, torch.float16),
            (4, 128, (16, 16), 32, torch.float16),
            # 2D spatial -- bf16
            (2, 64, (8, 8), 32, torch.bfloat16),
            (4, 128, (16, 16), 32, torch.bfloat16),
            # Different group counts
            (2, 64, (8, 8), 1, torch.float32),
            (2, 64, (8, 8), 64, torch.float32),
            (2, 64, (8, 8), 16, torch.float16),
            # Larger spatial
            (2, 32, (32, 32), 8, torch.float16),
        ]),
    ]


class GroupNormTest(TestBase):

    def __init__(self, N: int, C: int, spatial: tuple, G: int, dtype: torch.dtype, eps: float = 1e-5):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.G = G
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (self.N, self.C, *self.spatial)
        x = torch.randn(shape, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.C, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.C, dtype=self.dtype, device="cuda")
        return x, weight, bias

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x.float(),
            self.G,
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


@GroupNormFixture
def test_group_norm_op(N: int, C: int, spatial: tuple, G: int, dtype: torch.dtype) -> None:
    test = GroupNormTest(N, C, spatial, G, dtype)
    op = GroupNormOp(N=N, C=C, spatial=spatial, G=G, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class GroupNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("N, C, spatial, G, dtype", [
            (2, 64, (8, 8), 32, torch.float32),
            (2, 64, (8, 8), 32, torch.float16),
            (2, 64, (8, 8), 32, torch.bfloat16),
        ]),
    ]


@GroupNormNonContigFixture
def test_group_norm_non_contiguous(N: int, C: int, spatial: tuple, G: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    C_full = C * 2
    shape_full = (N, C_full, *spatial)
    x_full = torch.randn(shape_full, dtype=dtype, device="cuda")
    x = x_full[:, :C]  # non-contiguous slice
    weight = torch.randn(C, dtype=dtype, device="cuda")
    bias = torch.randn(C, dtype=dtype, device="cuda")

    op = GroupNormOp(N=N, C=C, spatial=spatial, G=G, dtype=dtype)

    y_ref = F.group_norm(
        x.contiguous().float(), G,
        weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
