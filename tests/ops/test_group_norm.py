import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.group_norm import GroupNormFwdOp, GroupNormFwdOpNoAffine
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
    op = GroupNormFwdOp(num_groups=g)
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

    op = GroupNormFwdOp(num_groups=g)

    y_ref = F.group_norm(
        x.contiguous().float(), g,
        weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


@pytest.mark.smoke
def test_group_norm_rejects_none_weight_or_bias() -> None:
    """Affine op rejects ``weight=None`` / ``bias=None``; affine-free path lives on NoAffine."""
    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    weight = torch.randn((c,), dtype=dtype, device="cuda")
    bias = torch.randn((c,), dtype=dtype, device="cuda")

    with pytest.raises((ValueError, TypeError)):
        op(x, None, bias)
    with pytest.raises((ValueError, TypeError)):
        op(x, weight, None)
    with pytest.raises((ValueError, TypeError)):
        op(x, None, None)


@pytest.mark.smoke
def test_group_norm_forward_required_signature() -> None:
    """`forward` declares weight and bias as required (no Optional, no default)."""
    import inspect
    sig = inspect.signature(GroupNormFwdOp.forward)
    weight_param = sig.parameters["weight"]
    bias_param = sig.parameters["bias"]
    assert weight_param.default is inspect.Parameter.empty
    assert bias_param.default is inspect.Parameter.empty


@pytest.mark.smoke
def test_group_norm_lazily_specializes_per_device() -> None:
    """A single op can lazily build specializations for different CUDA devices."""
    if torch.cuda.device_count() < 2:
        pytest.skip("multi-device test requires >= 2 CUDA devices")

    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x_other = torch.randn(
        (n, c, *spatial), dtype=dtype, device=torch.device("cuda", 1),
    )
    weight_other = torch.randn(
        (c,), dtype=dtype, device=torch.device("cuda", 1),
    )
    bias_other = torch.randn(
        (c,), dtype=dtype, device=torch.device("cuda", 1),
    )
    y = op(x_other, weight_other, bias_other)
    assert y.device == x_other.device
    assert len(op._kernel_cache) == 1


@pytest.mark.smoke
def test_group_norm_rejects_affine_device_mismatch() -> None:
    """Forward must raise ValueError when weight/bias live on a different CUDA device than x.

    Without an explicit check the kernel call would either dispatch on
    cross-device tensors (slow / wrong) or surface as an opaque CUDA
    error; surface a clean ValueError instead.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("affine-device-mismatch test requires >= 2 CUDA devices")

    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device=torch.device("cuda", 0))
    weight_other = torch.randn((c,), dtype=dtype, device=torch.device("cuda", 1))
    bias_other = torch.randn((c,), dtype=dtype, device=torch.device("cuda", 1))
    bias_same = torch.randn((c,), dtype=dtype, device=torch.device("cuda", 0))

    weight_same = torch.randn(
        (c,), dtype=dtype, device=torch.device("cuda", 0),
    )
    with pytest.raises(ValueError, match="weight on"):
        op(x, weight_other, bias_same)
    with pytest.raises(ValueError, match="bias on"):
        op(x, weight_same, bias_other)


class GroupNormNoAffineFixture(FixtureBase):
    PARAMS = [
        ("n, c, spatial, g, dtype", [
            pytest.param(2, 32, (8, 8), 8, torch.float32, marks=pytest.mark.smoke),
            pytest.param(2, 32, (8, 8), 8, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 32, (8, 8), 8, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4, 16, (4, 4), 4, torch.float16, marks=pytest.mark.full),
            # Non-aligned spatial: exercises padding path.
            pytest.param(2, 32, (7, 7), 8, torch.float16, marks=pytest.mark.full),
            # 1D spatial.
            pytest.param(2, 32, (16,), 8, torch.float16, marks=pytest.mark.full),
            # 3D spatial.
            pytest.param(2, 16, (4, 4, 4), 4, torch.float16, marks=pytest.mark.full),
        ]),
    ]


@GroupNormNoAffineFixture
def test_group_norm_no_affine_op(n: int, c: int, spatial: tuple, g: int,
                                 dtype: torch.dtype) -> None:
    """No-affine GroupNorm op matches torch.nn.functional.group_norm with weight=bias=None."""
    op = GroupNormFwdOpNoAffine(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    y = op(x)
    y_ref = F.group_norm(x.float(), g, weight=None, bias=None, eps=1e-5).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"max err: {(y - y_ref).abs().max()}"


@pytest.mark.smoke
def test_group_norm_no_affine_forward_signature() -> None:
    """No-affine forward accepts only x — no weight/bias parameters."""
    import inspect
    sig = inspect.signature(GroupNormFwdOpNoAffine.forward)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["x"], f"expected ['x'], got {params}"


@pytest.mark.smoke
def test_group_norm_no_affine_lazily_specializes_per_device() -> None:
    """No-affine op can lazily build specializations for different CUDA devices."""
    if torch.cuda.device_count() < 2:
        pytest.skip("multi-device test requires >= 2 CUDA devices")

    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOpNoAffine(num_groups=g)
    x_other = torch.randn(
        (n, c, *spatial), dtype=dtype, device=torch.device("cuda", 1),
    )
    y = op(x_other)
    assert y.device == x_other.device
    assert len(op._kernel_cache) == 1


@pytest.mark.smoke
def test_group_norm_no_affine_rejects_shape_mismatch() -> None:
    """Forward raises ValueError when input shape differs from configured (N, C, *spatial)."""
    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOpNoAffine(
        N=n, C=c, spatial=spatial, num_groups=g, dtype=dtype,
    )
    x_bad = torch.randn((n, c, 4, 8), dtype=dtype, device="cuda")
    with pytest.raises(ValueError, match="shape"):
        op(x_bad)


@pytest.mark.smoke
def test_group_norm_no_affine_rejects_dtype_mismatch() -> None:
    """Forward raises ValueError when input dtype differs from configured dtype."""
    n, c, spatial, g = 2, 32, (8, 8), 8
    op = GroupNormFwdOpNoAffine(
        N=n, C=c, spatial=spatial, num_groups=g, dtype=torch.float16,
    )
    x = torch.randn((n, c, *spatial), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        op(x)


@pytest.mark.smoke
@pytest.mark.parametrize("n, c, spatial, g", [
    # M = N * num_groups not divisible by max block_m (16): triggers tail
    # program reading/writing rows >= M before the M-padding fix.
    (1, 24, (4, 4), 3),   # M = 3
    (3, 30, (2, 2), 5),   # M = 15
    (1, 16, (8, 8), 1),   # M = 1
])
def test_group_norm_no_affine_tail_block(n: int, c: int, spatial: tuple,
                                         g: int) -> None:
    """No-affine GroupNorm handles M not divisible by the kernel's block_m."""
    dtype = torch.float16
    op = GroupNormFwdOpNoAffine(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    y = op(x)
    y_ref = F.group_norm(x.float(), g, weight=None, bias=None,
                        eps=1e-5).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
