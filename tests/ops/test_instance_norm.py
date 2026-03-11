import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.instance_norm import InstanceNormOp


class InstanceNormFixture(FixtureBase):
    PARAMS = [
        ("n, c, spatial, dtype, tune", [
            # Small CI-friendly shapes -- fp32
            pytest.param(2, 16, (8, 8), torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(4, 8, (4, 4), torch.float32, False, marks=pytest.mark.full),
            # Small CI-friendly shapes -- fp16
            pytest.param(2, 16, (8, 8), torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 8, (4, 4), torch.float16, False, marks=pytest.mark.full),
            # Small CI-friendly shapes -- bf16
            pytest.param(2, 16, (8, 8), torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(4, 8, (4, 4), torch.bfloat16, False, marks=pytest.mark.full),
            # 1D spatial
            pytest.param(2, 16, (16,), torch.float16, False, marks=pytest.mark.full),
            # 3D spatial
            pytest.param(2, 8, (4, 4, 4), torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


class InstanceNormTest(TestBase):

    def __init__(self, n: int, c: int, spatial: tuple,
                 dtype: torch.dtype, eps: float = 1e-5):
        self.n = n
        self.c = c
        self.spatial = spatial
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (self.n, self.c, *self.spatial)
        x = torch.randn(shape, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.c, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.c, dtype=self.dtype, device="cuda")
        return x, weight, bias

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor) -> torch.Tensor:
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
def test_instance_norm_op(n: int, c: int, spatial: tuple,
                          dtype: torch.dtype, tune: bool) -> None:
    test = InstanceNormTest(n, c, spatial, dtype)
    op = InstanceNormOp(N=n, C=c, spatial=spatial, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class InstanceNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("n, c, spatial, dtype", [
            pytest.param(2, 16, (8, 8), torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 16, (8, 8), torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@InstanceNormNonContigFixture
def test_instance_norm_non_contiguous(n: int, c: int, spatial: tuple,
                                      dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    shape = (n, c * 2, *spatial)
    x_full = torch.randn(shape, dtype=dtype, device="cuda")
    x = x_full[:, :c]  # non-contiguous slice
    weight = torch.randn(c, dtype=dtype, device="cuda")
    bias = torch.randn(c, dtype=dtype, device="cuda")

    op = InstanceNormOp(N=n, C=c, spatial=spatial, dtype=dtype)

    y_ref = F.instance_norm(
        x.contiguous().float(),
        weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
