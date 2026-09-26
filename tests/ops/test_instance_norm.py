import inspect

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase, standard_tolerance
from tileops.ops.norm.instance_norm import InstanceNormFwdOp
from workloads.normalization import InstanceNormWorkload


class InstanceNormTest(InstanceNormWorkload, TestBase):
    pass


class InstanceNormFixture(FixtureBase):
    PARAMS = [
        (
            "n, c, spatial, dtype, tune",
            [
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
            ],
        ),
    ]


@InstanceNormFixture
def test_instance_norm_op(n: int, c: int, spatial: tuple, dtype: torch.dtype, tune: bool) -> None:
    test = InstanceNormTest(n, c, spatial, dtype)
    op = InstanceNormFwdOp()
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


class InstanceNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "n, c, spatial, dtype",
            [
                pytest.param(2, 16, (8, 8), torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 16, (8, 8), torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@InstanceNormNonContigFixture
def test_instance_norm_non_contiguous(n: int, c: int, spatial: tuple, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    shape = (n, c * 2, *spatial)
    x_full = torch.randn(shape, dtype=dtype, device="cuda")
    x = x_full[:, :c]  # non-contiguous slice
    weight = torch.randn(c, dtype=dtype, device="cuda")
    bias = torch.randn(c, dtype=dtype, device="cuda")

    op = InstanceNormFwdOp()

    y_ref = F.instance_norm(
        x.contiguous().float(),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)

    y = op(x, weight=weight, bias=bias)
    assert torch.allclose(y, y_ref, **standard_tolerance(dtype)), (
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"
    )


class InstanceNormAffineFreeFixture(FixtureBase):
    PARAMS = [
        (
            "n, c, spatial, dtype, tune",
            [
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
            ],
        ),
    ]


@InstanceNormAffineFreeFixture
def test_instance_norm_affine_free_op(
    n: int, c: int, spatial: tuple, dtype: torch.dtype, tune: bool
) -> None:
    """Withholding the affine matches F.instance_norm(weight=None, bias=None)."""
    op = InstanceNormFwdOp()
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    y = op(x)
    y_ref = F.instance_norm(
        x.float(),
        weight=None,
        bias=None,
        eps=1e-5,
    ).to(dtype)
    assert torch.allclose(y, y_ref, **standard_tolerance(dtype)), (
        f"NoAffine forward mismatch, max err: {(y - y_ref).abs().max()}"
    )


@InstanceNormAffineFreeFixture
def test_instance_norm_affine_free_running_stats(
    n: int,
    c: int,
    spatial: tuple,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """use_input_stats=False uses running_mean/running_var; matches torch reference."""
    op = InstanceNormFwdOp(use_input_stats=False)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    running_mean = torch.randn(c, dtype=torch.float32, device="cuda")
    running_var = torch.rand(c, dtype=torch.float32, device="cuda") + 0.1
    y = op(x, running_mean, running_var)
    y_ref = F.instance_norm(
        x,
        running_mean=running_mean,
        running_var=running_var,
        weight=None,
        bias=None,
        use_input_stats=False,
        eps=1e-5,
    )
    assert torch.allclose(y, y_ref, **standard_tolerance(dtype)), (
        f"Running-stats mismatch, max err: {(y - y_ref).abs().max()}"
    )


@pytest.mark.smoke
def test_instance_norm_validate_dtypes_matches_manifest_inputs() -> None:
    """``_validate_dtypes`` accepts kwargs matching manifest ``signature.inputs``.

    Regression guard for a signature drift where the hand-written override
    accepted only ``x`` while the manifest declared ``x``, ``weight`` and
    ``bias``. The manifest-validator dtype-parity check binds by kwargs and
    requires the impl to honor the manifest order.
    """
    sig = inspect.signature(InstanceNormFwdOp._validate_dtypes)
    params = [p for p in sig.parameters if p != "self"]
    expected = ["x", "running_mean", "running_var", "weight", "bias"]
    assert params == expected, (
        f"_validate_dtypes params {params} must match manifest inputs {expected} in order"
    )


@pytest.mark.smoke
def test_instance_norm_lazily_specializes_per_device() -> None:
    """An op first called on a non-default CUDA device builds its entry there."""
    if torch.cuda.device_count() < 2:
        pytest.skip("multi-device test requires >= 2 CUDA devices")

    n, c, spatial, dtype = 2, 32, (8, 8), torch.float16
    op = InstanceNormFwdOp()
    x_other = torch.randn(
        (n, c, *spatial),
        dtype=dtype,
        device=torch.device("cuda", 1),
    )
    weight_other = torch.randn(
        (c,),
        dtype=dtype,
        device=torch.device("cuda", 1),
    )
    bias_other = torch.randn(
        (c,),
        dtype=dtype,
        device=torch.device("cuda", 1),
    )
    y = op(x_other, weight=weight_other, bias=bias_other)
    assert y.device == x_other.device
    assert len(op.built_kernels("instance_norm")) == 1


@pytest.mark.smoke
def test_instance_norm_lazy_cache_reuse_and_respecialization() -> None:
    """One op instance reuses identical specs and caches changed specs."""
    op = InstanceNormFwdOp()

    def run_case(n: int, c: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
        weight = torch.randn((c,), dtype=dtype, device="cuda")
        bias = torch.randn((c,), dtype=dtype, device="cuda")

        y = op(x, weight=weight, bias=bias)
        y_ref = F.instance_norm(
            x.float(),
            weight=weight.float(),
            bias=bias.float(),
            eps=1e-5,
        ).to(dtype)
        assert torch.allclose(y, y_ref, **standard_tolerance(dtype))

    run_case(2, 8, (4, 4), torch.float16)
    assert len(op.built_kernels("instance_norm")) == 1
    assert op.eval_roofline() == (
        5 * 2 * 8 * 16,
        (2 * 2 * 8 * 16 + 2 * 8) * torch.float16.itemsize,
    )

    run_case(2, 8, (4, 4), torch.float16)
    assert len(op.built_kernels("instance_norm")) == 1

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert len(op.built_kernels("instance_norm")) == 2
    assert op.eval_roofline() == (
        5 * 3 * 12 * 16,
        (2 * 3 * 12 * 16 + 2 * 12) * torch.bfloat16.itemsize,
    )


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
        op = InstanceNormFwdOp()
    x = torch.randn((n, c, *spatial), dtype=dtype, device=torch.device("cuda", 0))
    weight_other = torch.randn((c,), dtype=dtype, device=torch.device("cuda", 1))
    bias_other = torch.randn((c,), dtype=dtype, device=torch.device("cuda", 1))
    bias_same = torch.randn((c,), dtype=dtype, device=torch.device("cuda", 0))

    weight_same = torch.randn(
        (c,),
        dtype=dtype,
        device=torch.device("cuda", 0),
    )
    with pytest.raises(ValueError, match="weight on"):
        op(x, weight=weight_other, bias=bias_same)
    with pytest.raises(ValueError, match="bias on"):
        op(x, weight=weight_same, bias=bias_other)


_OP_CLASSES = [
    pytest.param(InstanceNormFwdOp, "InstanceNormFwdOp", id="InstanceNormFwdOp"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, manifest_key", _OP_CLASSES)
def test_instance_norm_init_accepts_use_input_stats_and_momentum(
    op_cls: type,
    manifest_key: str,
) -> None:
    """`__init__` must expose the manifest-declared params so L1 parity holds.

    The manifest entry declares `use_input_stats` and `momentum` (matching
    PyTorch's `torch.nn.functional.instance_norm` public API). The op must
    accept both, defaulting to PyTorch's defaults.
    """
    init_params = inspect.signature(op_cls.__init__).parameters
    assert "use_input_stats" in init_params
    assert "momentum" in init_params
    assert init_params["use_input_stats"].default is True
    assert init_params["momentum"].default == pytest.approx(0.1)


@pytest.mark.smoke
def test_instance_norm_default_momentum_does_not_change_output() -> None:
    """Per-batch path is independent of `momentum`; default value must match torch."""
    n, c, spatial, dtype = 2, 16, (8, 8), torch.float16
    op_default = InstanceNormFwdOp()
    op_other = InstanceNormFwdOp(momentum=0.5)
    assert op_default.momentum == pytest.approx(0.1)
    assert op_other.momentum == pytest.approx(0.5)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    weight = torch.randn((c,), dtype=dtype, device="cuda")
    bias = torch.randn((c,), dtype=dtype, device="cuda")
    y1 = op_default(x, weight=weight, bias=bias)
    y2 = op_other(x, weight=weight, bias=bias)
    assert torch.allclose(y1, y2, **standard_tolerance(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "use_input_stats, affine",
    [(True, "weight"), (True, "bias"), (False, "both"), (True, "none")],
)
def test_instance_norm_matches_torch_on_every_presence_branch(use_input_stats, affine) -> None:
    """Affine tensors are independent, and running statistics are read in eval and updated
    in the input's dtype otherwise, as ``torch.nn.functional.instance_norm`` does."""
    c, dtype = 16, torch.float16
    x = torch.randn((2, c, 8, 8), dtype=dtype, device="cuda") * 3 + 1
    weight = torch.randn(c, dtype=dtype, device="cuda") if affine in ("weight", "both") else None
    bias = torch.randn(c, dtype=dtype, device="cuda") if affine in ("bias", "both") else None
    stats = (torch.randn(c, device="cuda"), torch.rand(c, device="cuda") + 0.5)
    mine, ref = [s.clone() for s in stats], [s.clone() for s in stats]
    op = InstanceNormFwdOp(use_input_stats=use_input_stats)
    y = op(x, *mine, weight, bias)
    y_ref = F.instance_norm(x, *ref, weight, bias, use_input_stats=use_input_stats)
    torch.testing.assert_close(y, y_ref, **standard_tolerance(dtype))
    torch.testing.assert_close(mine, ref, **standard_tolerance(dtype))


@pytest.mark.smoke
def test_instance_norm_needs_both_running_statistics_to_read_them() -> None:
    x = torch.randn((2, 16, 8, 8), dtype=torch.float16, device="cuda")
    stat = torch.zeros(16, device="cuda")
    with pytest.raises(ValueError, match="use_input_stats or present"):
        InstanceNormFwdOp(use_input_stats=False)(x)
    with pytest.raises(ValueError, match="'running_var' is required"):
        InstanceNormFwdOp()(x, stat)
