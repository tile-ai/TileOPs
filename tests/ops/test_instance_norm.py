import inspect

import pytest
import torch
import torch.nn.functional as F
import yaml

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.instance_norm import InstanceNormFwdOp
from workloads.normalization import InstanceNormWorkload


class InstanceNormTest(InstanceNormWorkload, TestBase):
    pass


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
    op = InstanceNormFwdOp()
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

    op = InstanceNormFwdOp()

    y_ref = F.instance_norm(
        x.contiguous().float(),
        weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)

    y = op(x, weight=weight, bias=bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


class InstanceNormAffineFreeFixture(FixtureBase):
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


@InstanceNormAffineFreeFixture
def test_instance_norm_affine_free_op(n: int, c: int, spatial: tuple,
                                    dtype: torch.dtype, tune: bool) -> None:
    """Withholding the affine matches F.instance_norm(weight=None, bias=None)."""
    op = InstanceNormFwdOp()
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    y = op(x)
    y_ref = F.instance_norm(
        x.float(), weight=None, bias=None, eps=1e-5,
    ).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"NoAffine forward mismatch, max err: {(y - y_ref).abs().max()}"


@InstanceNormAffineFreeFixture
def test_instance_norm_affine_free_running_stats(
    n: int, c: int, spatial: tuple, dtype: torch.dtype, tune: bool,
) -> None:
    """use_input_stats=False uses running_mean/running_var; matches torch reference."""
    op = InstanceNormFwdOp(use_input_stats=False)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    running_mean = torch.randn(c, dtype=torch.float32, device="cuda")
    running_var = torch.rand(c, dtype=torch.float32, device="cuda") + 0.1
    y = op(x, running_mean, running_var)
    y_ref = F.instance_norm(
        x, running_mean=running_mean, running_var=running_var,
        weight=None, bias=None, use_input_stats=False, eps=1e-5,
    )
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"Running-stats mismatch, max err: {(y - y_ref).abs().max()}"


@pytest.mark.smoke
def test_instance_norm_rejects_half_a_switch() -> None:
    """weight and bias move together, and so do the running stats."""
    n, c, spatial, dtype = 2, 16, (8, 8), torch.float16
    op = InstanceNormFwdOp()
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    weight = torch.randn((c,), dtype=dtype, device="cuda")
    stat = torch.zeros((c,), dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="one switch"):
        op(x, weight=weight)
    with pytest.raises(ValueError, match="one switch"):
        op(x, bias=weight)
    with pytest.raises(ValueError, match="one switch"):
        op(x, running_mean=stat)


@pytest.mark.smoke
def test_instance_norm_rejects_input_affine_dtype_mismatch() -> None:
    op = InstanceNormFwdOp.__new__(InstanceNormFwdOp)

    fp16 = torch.empty(0, dtype=torch.float16)
    bf16 = torch.empty(0, dtype=torch.bfloat16)
    int32 = torch.empty(0, dtype=torch.int32)

    op._validate_dtypes(fp16, weight=fp16, bias=fp16)

    with pytest.raises(ValueError, match="x.dtype"):
        op._validate_dtypes(int32, weight=fp16, bias=fp16)
    with pytest.raises(ValueError, match="weight.dtype"):
        op._validate_dtypes(fp16, weight=bf16, bias=fp16)
    with pytest.raises(ValueError, match="bias.dtype"):
        op._validate_dtypes(fp16, weight=fp16, bias=bf16)


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
        f"_validate_dtypes params {params} must match manifest inputs "
        f"{expected} in order"
    )


@pytest.mark.smoke
def test_instance_norm_lazily_specializes_per_device() -> None:
    """A single op can lazily build specializations for different CUDA devices."""
    if torch.cuda.device_count() < 2:
        pytest.skip("multi-device test requires >= 2 CUDA devices")

    n, c, spatial, dtype = 2, 32, (8, 8), torch.float16
    op = InstanceNormFwdOp()
    x_other = torch.randn(
        (n, c, *spatial), dtype=dtype, device=torch.device("cuda", 1),
    )
    weight_other = torch.randn(
        (c,), dtype=dtype, device=torch.device("cuda", 1),
    )
    bias_other = torch.randn(
        (c,), dtype=dtype, device=torch.device("cuda", 1),
    )
    y = op(x_other, weight=weight_other, bias=bias_other)
    assert y.device == x_other.device
    assert len(list(op.iter_kernels())) == 1


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
            x.float(), weight=weight.float(), bias=bias.float(), eps=1e-5,
        ).to(dtype)
        atol, rtol = _get_tolerances(dtype)
        assert torch.allclose(y, y_ref, atol=atol, rtol=rtol)

    run_case(2, 8, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1
    assert op.eval_roofline() == (
        5 * 2 * 8 * 16,
        (2 * 2 * 8 * 16 + 2 * 8) * torch.float16.itemsize,
    )

    run_case(2, 8, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert len(list(op.iter_kernels())) == 2
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
        (c,), dtype=dtype, device=torch.device("cuda", 0),
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
    op_cls: type, manifest_key: str,
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
@pytest.mark.parametrize("op_cls, manifest_key", _OP_CLASSES)
def test_instance_norm_init_signature_covers_manifest_params(
    op_cls: type, manifest_key: str,
) -> None:
    """Union of `__init__` and `forward` params must cover manifest params."""
    from pathlib import Path

    manifest_file = (
        Path(__file__).resolve().parents[2]
        / "src" / "tileops" / "manifest" / "normalization.yaml"
    )
    with open(manifest_file) as fp:
        manifest = yaml.safe_load(fp) or {}
    manifest_params = set(
        manifest[manifest_key]["signature"]["params"].keys()
    )
    init_params = set(inspect.signature(op_cls.__init__).parameters)
    forward_params = set(inspect.signature(op_cls.forward).parameters)
    code_params = (init_params | forward_params) - {"self"}
    missing = manifest_params - code_params
    assert not missing, f"manifest params not covered by code: {missing}"


@pytest.mark.smoke
def test_instance_norm_batch_stats_path_rejects_running_stats() -> None:
    """They normalize on the eval path only, and no path updates them."""
    c = 16
    op = InstanceNormFwdOp()
    x = torch.randn((2, c, 8, 8), dtype=torch.float32, device="cuda")
    stat = torch.randn(c, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="use_input_stats=False"):
        op(x, stat, stat.abs() + 0.1)


@pytest.mark.smoke
def test_instance_norm_running_stats_path_rejects_affine() -> None:
    """The affine variant still defers `use_input_stats=False`."""
    op = InstanceNormFwdOp(use_input_stats=False)
    x = torch.randn((2, 16, 8, 8), dtype=torch.float16, device="cuda")
    stat = torch.zeros((16,), dtype=torch.float32, device="cuda")
    affine = torch.randn((16,), dtype=torch.float16, device="cuda")
    with pytest.raises(NotImplementedError, match="affine-free"):
        op(x, stat, stat + 1, affine, affine)
    with pytest.raises(ValueError, match="running_mean and running_var must"):
        op(x)


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
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
