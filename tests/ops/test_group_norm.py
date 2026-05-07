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
            pytest.param(4, 16, (4, 4), 4, torch.float32, False, marks=pytest.mark.full),
            pytest.param(4, 16, (4, 4), 4, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 16, (4, 4), 4, torch.bfloat16, False, marks=pytest.mark.full),
            # Different group counts
            pytest.param(2, 32, (4, 4), 1, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 32, (4, 4), 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 32, (4, 4), 16, torch.float16, False, marks=pytest.mark.full),
            # 1D spatial
            pytest.param(2, 32, (16,), 8, torch.float16, False, marks=pytest.mark.full),
            # 3D spatial
            pytest.param(2, 16, (4, 4, 4), 4, torch.float16, False, marks=pytest.mark.full),
            # Non-power-of-two channels per group
            pytest.param(2, 30, (4, 4), 5, torch.float16, False, marks=pytest.mark.full),
            # Non-aligned spatial: exercises partial-tile path
            pytest.param(2, 32, (7, 7), 8, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 32, (7, 7), 8, torch.bfloat16, False, marks=pytest.mark.full),
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
    op = GroupNormFwdOp(N=n, C=c, spatial=spatial, num_groups=g, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class GroupNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("n, c, spatial, g, dtype", [
            pytest.param(2, 32, (8, 8), 8, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 32, (8, 8), 8, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@GroupNormNonContigFixture
def test_group_norm_non_contiguous(n: int, c: int, spatial: tuple, g: int,
                                   dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    shape = (n, c * 2, *spatial)
    x_full = torch.randn(shape, dtype=dtype, device="cuda")
    x = x_full[:, :c]  # non-contiguous slice
    weight = torch.randn(c, dtype=dtype, device="cuda")
    bias = torch.randn(c, dtype=dtype, device="cuda")

    op = GroupNormFwdOp(N=n, C=c, spatial=spatial, num_groups=g, dtype=dtype)

    y_ref = F.group_norm(
        x.contiguous().float(), g,
        weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


@pytest.mark.smoke
def test_group_norm_optional_weight_bias_cache_stable() -> None:
    """When weight/bias are None, repeated forwards reuse cached affine tensors."""
    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(N=n, C=c, spatial=spatial, num_groups=g, dtype=dtype)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")

    y1 = op(x, None, None)
    cached_weight_id = id(op._cached_unit_weight)
    cached_bias_id = id(op._cached_zero_bias)

    y2 = op(x, None, None)
    assert id(op._cached_unit_weight) == cached_weight_id, \
        "unit_weight cache should be reused across forward calls"
    assert id(op._cached_zero_bias) == cached_bias_id, \
        "zero_bias cache should be reused across forward calls"

    # Correctness: matches torch reference (weight=None / bias=None).
    y_ref = F.group_norm(x.float(), g, weight=None, bias=None, eps=1e-5).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y1, y_ref, atol=atol, rtol=rtol)
    assert torch.allclose(y2, y_ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_group_norm_cache_rebuilds_on_dtype_change() -> None:
    """Cache rebuilds when input dtype changes; cached tensors track input dtype."""
    n, c, spatial, g = 2, 32, (8, 8), 8

    # First combo: float16
    op16 = GroupNormFwdOp(N=n, C=c, spatial=spatial, num_groups=g,
                          dtype=torch.float16)
    x16 = torch.randn((n, c, *spatial), dtype=torch.float16, device="cuda")
    op16(x16, None, None)
    assert op16._cached_unit_weight.dtype == torch.float16
    assert op16._cached_unit_weight.device == x16.device

    # Second combo: bfloat16 (different dtype, same device class)
    op_bf = GroupNormFwdOp(N=n, C=c, spatial=spatial, num_groups=g,
                           dtype=torch.bfloat16)
    x_bf = torch.randn((n, c, *spatial), dtype=torch.bfloat16, device="cuda")
    op_bf(x_bf, None, None)
    assert op_bf._cached_unit_weight.dtype == torch.bfloat16
    assert op_bf._cached_zero_bias.dtype == torch.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
