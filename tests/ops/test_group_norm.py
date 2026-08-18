import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.group_norm import GroupNormFwdOp
from workloads.normalization import GroupNormWorkload


class GroupNormTest(GroupNormWorkload, TestBase):
    pass


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
def test_group_norm_no_affine_matches_torch() -> None:
    """Omitting the affine pair is the torch.nn.GroupNorm(affine=False) path."""
    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    y = op(x)
    y_ref = F.group_norm(x.float(), g, weight=None, bias=None,
                         eps=1e-5).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"max err: {(y - y_ref).abs().max()}"


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
    assert len(list(op.iter_kernels())) == 1


@pytest.mark.smoke
def test_group_norm_lazy_cache_reuse_and_respecialization() -> None:
    """One op instance reuses identical specs and caches changed specs."""
    op = GroupNormFwdOp(num_groups=4)

    def run_case(n: int, c: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
        weight = torch.randn((c,), dtype=dtype, device="cuda")
        bias = torch.randn((c,), dtype=dtype, device="cuda")

        y = op(x, weight, bias)
        y_ref = F.group_norm(
            x.float(), 4, weight=weight.float(), bias=bias.float(), eps=1e-5,
        ).to(dtype)
        atol, rtol = _get_tolerances(dtype)
        assert torch.allclose(y, y_ref, atol=atol, rtol=rtol)

    run_case(2, 16, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1
    assert op.eval_roofline() == (
        5 * 2 * 16 * 16,
        (2 * 2 * 16 * 16 + 2 * 16) * torch.float16.itemsize,
    )

    run_case(2, 16, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1

    run_case(3, 24, (2, 8), torch.bfloat16)
    assert len(list(op.iter_kernels())) == 2
    assert op.eval_roofline() == (
        5 * 3 * 24 * 16,
        (2 * 3 * 24 * 16 + 2 * 24) * torch.bfloat16.itemsize,
    )


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
    op = GroupNormFwdOp(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    y = op(x)
    y_ref = F.group_norm(x.float(), g, weight=None, bias=None, eps=1e-5).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"max err: {(y - y_ref).abs().max()}"


@pytest.mark.smoke
def test_group_norm_forward_signature() -> None:
    """One forward takes x plus the optional affine pair (R18)."""
    import inspect
    sig = inspect.signature(GroupNormFwdOp.forward)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["x", "weight", "bias"], f"got {params}"
    for name in ("weight", "bias"):
        assert sig.parameters[name].default is None, \
            f"{name} must default to None"


@pytest.mark.smoke
@pytest.mark.parametrize("give", ["weight", "bias"])
def test_group_norm_rejects_half_the_affine_switch(give: str) -> None:
    """weight and bias are one switch; half of it is an error (R18.3)."""
    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    t = torch.randn((c,), dtype=dtype, device="cuda")
    kwargs = {give: t}
    with pytest.raises(ValueError, match="one switch"):
        op(x, **kwargs)


@pytest.mark.smoke
def test_group_norm_no_affine_lazily_specializes_per_device() -> None:
    """No-affine op can lazily build specializations for different CUDA devices."""
    if torch.cuda.device_count() < 2:
        pytest.skip("multi-device test requires >= 2 CUDA devices")

    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x_other = torch.randn(
        (n, c, *spatial), dtype=dtype, device=torch.device("cuda", 1),
    )
    y = op(x_other)
    assert y.device == x_other.device
    assert len(list(op.iter_kernels())) == 1


@pytest.mark.smoke
@pytest.mark.parametrize("n, c, spatial, g", [
    # Very few rows, so the grid is one or two blocks wide.
    (1, 24, (4, 4), 3),   # M = 3
    (3, 30, (2, 2), 5),   # M = 15
    (1, 16, (8, 8), 1),   # M = 1
])
def test_group_norm_no_affine_tail_block(n: int, c: int, spatial: tuple,
                                         g: int) -> None:
    """No-affine GroupNorm handles a row count smaller than one grid block."""
    dtype = torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    y = op(x)
    y_ref = F.group_norm(x.float(), g, weight=None, bias=None,
                        eps=1e-5).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
