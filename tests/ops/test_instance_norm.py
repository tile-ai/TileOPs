import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.instance_norm import (
    InstanceNormFwdOp,
    InstanceNormFwdOpNoAffine,
)
from workloads.instance_norm import InstanceNormTest as _InstanceNormTestWorkload


class InstanceNormTest(_InstanceNormTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor) -> torch.Tensor:
        return F.instance_norm(
            x.float(),
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)


class InstanceNormFixture(FixtureBase):
    PARAMS = [
        ("n, c, spatial, dtype, tune", [
            # Small CI-friendly shapes -- fp32
            pytest.param(2, 16, (8, 8), torch.float32, False, marks=pytest.mark.smoke),
            # Small CI-friendly shapes -- fp16
            pytest.param(2, 16, (8, 8), torch.float16, False, marks=pytest.mark.smoke),
            # Small CI-friendly shapes -- bf16
            pytest.param(2, 16, (8, 8), torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(4, 8, (4, 4), torch.float32, False, marks=pytest.mark.full),
            pytest.param(4, 8, (4, 4), torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 8, (4, 4), torch.bfloat16, False, marks=pytest.mark.full),
            # 1D spatial
            pytest.param(2, 16, (16,), torch.float16, False, marks=pytest.mark.full),
            # 3D spatial
            pytest.param(2, 8, (4, 4, 4), torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


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
    op = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class InstanceNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("n, c, spatial, dtype", [
            pytest.param(2, 16, (8, 8), torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 16, (8, 8), torch.bfloat16, marks=pytest.mark.smoke),
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

    op = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=dtype)

    y_ref = F.instance_norm(
        x.contiguous().float(),
        weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


@pytest.mark.smoke
def test_instance_norm_optional_weight_bias_cache_stable() -> None:
    """When weight/bias are None, repeated forwards reuse cached affine tensors."""
    n, c, spatial, dtype = 2, 16, (8, 8), torch.float16
    op = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=dtype)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")

    y1 = op(x, None, None)
    cached_weight_id = id(op._cached_unit_weight)
    cached_bias_id = id(op._cached_zero_bias)

    y2 = op(x, None, None)
    assert id(op._cached_unit_weight) == cached_weight_id, \
        "unit_weight cache should be reused across forward calls"
    assert id(op._cached_zero_bias) == cached_bias_id, \
        "zero_bias cache should be reused across forward calls"

    y_ref = F.instance_norm(x.float(), weight=None, bias=None, eps=1e-5).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y1, y_ref, atol=atol, rtol=rtol)
    assert torch.allclose(y2, y_ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_instance_norm_cache_rebuilds_on_dtype_change() -> None:
    """Cache rebuilds when input dtype changes; cached tensors track input dtype."""
    n, c, spatial = 2, 16, (8, 8)

    op16 = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=torch.float16)
    x16 = torch.randn((n, c, *spatial), dtype=torch.float16, device="cuda")
    op16(x16, None, None)
    assert op16._cached_unit_weight.dtype == torch.float16
    assert op16._cached_unit_weight.device == x16.device

    op_bf = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=torch.bfloat16)
    x_bf = torch.randn((n, c, *spatial), dtype=torch.bfloat16, device="cuda")
    op_bf(x_bf, None, None)
    assert op_bf._cached_unit_weight.dtype == torch.bfloat16
    assert op_bf._cached_zero_bias.dtype == torch.bfloat16


@pytest.mark.smoke
def test_instance_norm_supplied_affine_does_not_consult_cache() -> None:
    """When weight and bias are both supplied, the cache must not be consulted."""
    n, c, spatial, dtype = 2, 32, (8, 8), torch.float16
    op = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=dtype)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    weight = torch.randn((c,), dtype=dtype, device="cuda")
    bias = torch.randn((c,), dtype=dtype, device="cuda")

    def _raise(*args, **kwargs):
        raise AssertionError(
            "_get_affine_identity must not be called on the supplied-affine path"
        )

    op._get_affine_identity = _raise  # type: ignore[method-assign]

    y = op(x, weight, bias)

    # Correctness sanity check: matches torch reference.
    y_ref = F.instance_norm(
        x.float(), weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol)

    # Cache state must remain untouched.
    assert op._cached_unit_weight is None
    assert op._cached_zero_bias is None
    assert op._cached_affine_key is None


@pytest.mark.smoke
def test_instance_norm_rejects_device_mismatch() -> None:
    """Forward must raise ValueError when input device differs from kernel device.

    The compiled kernel binds to the active CUDA device at construction
    time, so callers must construct one op per device. Verify the op
    surfaces a clean ValueError rather than letting the kernel layer
    raise an opaque device-id error.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("device-mismatch test requires >= 2 CUDA devices")

    n, c, spatial, dtype = 2, 32, (8, 8), torch.float16
    with torch.cuda.device(0):
        op = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=dtype)
    x_other = torch.randn(
        (n, c, *spatial), dtype=dtype, device=torch.device("cuda", 1),
    )
    with pytest.raises(ValueError, match="[Dd]evice mismatch"):
        op(x_other, None, None)


@pytest.mark.smoke
def test_instance_norm_rejects_affine_device_mismatch() -> None:
    """Forward must raise ValueError when weight/bias live on a different CUDA device than x.

    Without an explicit check the kernel call would either dispatch on
    cross-device tensors (slow / wrong) or surface as an opaque CUDA
    error; surface a clean ValueError instead.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("affine-device-mismatch test requires >= 2 CUDA devices")

    n, c, spatial, dtype = 2, 32, (8, 8), torch.float16
    with torch.cuda.device(0):
        op = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=dtype)
    x = torch.randn((n, c, *spatial), dtype=dtype, device=torch.device("cuda", 0))
    weight_other = torch.randn((c,), dtype=dtype, device=torch.device("cuda", 1))
    bias_other = torch.randn((c,), dtype=dtype, device=torch.device("cuda", 1))
    bias_same = torch.randn((c,), dtype=dtype, device=torch.device("cuda", 0))

    with pytest.raises(ValueError, match="weight on"):
        op(x, weight_other, bias_same)
    with pytest.raises(ValueError, match="bias on"):
        op(x, None, bias_other)


# FIXME(staged-rollout): skip no-affine InstanceNorm tests while the
# manifest entry is `status: spec-only`.
#
# Broken invariant: the manifest declares `use_input_stats` and `momentum`
# (matching torch.nn.functional.instance_norm), but the current op only
# implements the `use_input_stats=True` branch and lacks running_mean /
# running_var arguments — so the implementation does not yet conform to
# the spec.
# Why: trust model requires status=spec-only when implementation is
# incomplete; the running-stats path lands in a follow-up PR.
# Cleanup: remove this marker once `InstanceNormFwdOpNoAffine` flips back
# to `status: implemented` with full running-stats support.
_NO_AFFINE_PENDING_RUNNING_STATS = pytest.mark.skip(
    reason="InstanceNormFwdOpNoAffine is spec-only pending running-stats path",
)


class InstanceNormNoAffineFixture(FixtureBase):
    PARAMS = [
        ("n, c, spatial, dtype", [
            pytest.param(2, 16, (8, 8), torch.float32, marks=pytest.mark.smoke),
            pytest.param(2, 16, (8, 8), torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 16, (8, 8), torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4, 8, (4, 4), torch.float16, marks=pytest.mark.full),
            # 1D spatial.
            pytest.param(2, 16, (16,), torch.float16, marks=pytest.mark.full),
            # 3D spatial.
            pytest.param(2, 8, (4, 4, 4), torch.float16, marks=pytest.mark.full),
        ]),
    ]


@_NO_AFFINE_PENDING_RUNNING_STATS
@InstanceNormNoAffineFixture
def test_instance_norm_no_affine_op(n: int, c: int, spatial: tuple,
                                    dtype: torch.dtype) -> None:
    """No-affine InstanceNorm op matches torch.nn.functional.instance_norm with weight=bias=None."""
    op = InstanceNormFwdOpNoAffine(N=n, C=c, spatial=spatial, dtype=dtype)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    y = op(x)
    y_ref = F.instance_norm(x.float(), weight=None, bias=None, eps=1e-5).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"max err: {(y - y_ref).abs().max()}"


@_NO_AFFINE_PENDING_RUNNING_STATS
@pytest.mark.smoke
def test_instance_norm_no_affine_forward_signature() -> None:
    """No-affine forward accepts only x — no weight/bias parameters."""
    import inspect
    sig = inspect.signature(InstanceNormFwdOpNoAffine.forward)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["x"], f"expected ['x'], got {params}"


@_NO_AFFINE_PENDING_RUNNING_STATS
@pytest.mark.smoke
def test_instance_norm_no_affine_rejects_device_mismatch() -> None:
    """Forward raises ValueError when input lives on a different CUDA device."""
    if torch.cuda.device_count() < 2:
        pytest.skip("device-mismatch test requires >= 2 CUDA devices")

    n, c, spatial, dtype = 2, 16, (8, 8), torch.float16
    with torch.cuda.device(0):
        op = InstanceNormFwdOpNoAffine(N=n, C=c, spatial=spatial, dtype=dtype)
    x_other = torch.randn(
        (n, c, *spatial), dtype=dtype, device=torch.device("cuda", 1),
    )
    with pytest.raises(ValueError, match="[Dd]evice mismatch"):
        op(x_other)


@_NO_AFFINE_PENDING_RUNNING_STATS
@pytest.mark.smoke
def test_instance_norm_no_affine_rejects_shape_mismatch() -> None:
    """Forward raises ValueError when input shape differs from configured (N, C, *spatial)."""
    n, c, spatial, dtype = 2, 16, (8, 8), torch.float16
    op = InstanceNormFwdOpNoAffine(N=n, C=c, spatial=spatial, dtype=dtype)
    x_bad = torch.randn((n, c, 4, 8), dtype=dtype, device="cuda")
    with pytest.raises(ValueError, match="shape"):
        op(x_bad)


@_NO_AFFINE_PENDING_RUNNING_STATS
@pytest.mark.smoke
def test_instance_norm_no_affine_rejects_use_input_stats_false() -> None:
    """Constructor raises ValueError when use_input_stats=False (unsupported)."""
    with pytest.raises(ValueError, match="use_input_stats"):
        InstanceNormFwdOpNoAffine(
            N=2, C=16, spatial=(8, 8), dtype=torch.float16,
            use_input_stats=False,
        )


@_NO_AFFINE_PENDING_RUNNING_STATS
@pytest.mark.smoke
def test_instance_norm_no_affine_rejects_dtype_mismatch() -> None:
    """Forward raises ValueError when input dtype differs from configured dtype."""
    n, c, spatial = 2, 16, (8, 8)
    op = InstanceNormFwdOpNoAffine(
        N=n, C=c, spatial=spatial, dtype=torch.float16,
    )
    x = torch.randn((n, c, *spatial), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        op(x)


@_NO_AFFINE_PENDING_RUNNING_STATS
@pytest.mark.smoke
@pytest.mark.parametrize("n, c, spatial", [
    # M = N * C not divisible by max block_m (16): triggers tail program
    # reading/writing rows >= M before the M-padding fix.
    (1, 3, (4, 4)),    # M = 3
    (3, 5, (2, 2)),    # M = 15
    (1, 1, (8, 8)),    # M = 1
])
def test_instance_norm_no_affine_tail_block(n: int, c: int,
                                            spatial: tuple) -> None:
    """No-affine InstanceNorm handles M not divisible by the kernel's block_m."""
    dtype = torch.float16
    op = InstanceNormFwdOpNoAffine(N=n, C=c, spatial=spatial, dtype=dtype)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    y = op(x)
    y_ref = F.instance_norm(x.float(), weight=None, bias=None,
                            eps=1e-5).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
