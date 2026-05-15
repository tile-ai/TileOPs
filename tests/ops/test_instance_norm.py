import inspect

import pytest
import torch
import torch.nn.functional as F
import yaml

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


class InstanceNormNoAffineFixture(FixtureBase):
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


@InstanceNormNoAffineFixture
def test_instance_norm_no_affine_op(n: int, c: int, spatial: tuple,
                                    dtype: torch.dtype, tune: bool) -> None:
    """Forward correctness for InstanceNormFwdOpNoAffine vs F.instance_norm(weight=None, bias=None)."""
    op = InstanceNormFwdOpNoAffine(N=n, C=c, spatial=spatial, dtype=dtype)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    # Running stats are required positional args (R16) but ignored on the
    # use_input_stats=True path; pass placeholders.
    rm = torch.zeros(c, dtype=torch.float32, device="cuda")
    rv = torch.ones(c, dtype=torch.float32, device="cuda")
    y = op(x, rm, rv)
    y_ref = F.instance_norm(
        x.float(), weight=None, bias=None, eps=1e-5,
    ).to(dtype)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"NoAffine forward mismatch, max err: {(y - y_ref).abs().max()}"


@InstanceNormNoAffineFixture
def test_instance_norm_no_affine_running_stats(
    n: int, c: int, spatial: tuple, dtype: torch.dtype, tune: bool,
) -> None:
    """use_input_stats=False uses running_mean/running_var; matches torch reference."""
    op = InstanceNormFwdOpNoAffine(
        N=n, C=c, spatial=spatial, dtype=dtype, use_input_stats=False,
    )
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
        f"NoAffine running-stats mismatch, max err: {(y - y_ref).abs().max()}"


@pytest.mark.smoke
def test_instance_norm_rejects_none_weight_or_bias() -> None:
    """Affine op rejects ``weight=None`` / ``bias=None``; affine-free path lives on NoAffine."""
    n, c, spatial, dtype = 2, 16, (8, 8), torch.float16
    op = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=dtype)
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
def test_instance_norm_forward_required_signature() -> None:
    """`forward` declares weight and bias as required (no Optional, no default)."""
    sig = inspect.signature(InstanceNormFwdOp.forward)
    weight_param = sig.parameters["weight"]
    bias_param = sig.parameters["bias"]
    assert weight_param.default is inspect.Parameter.empty
    assert bias_param.default is inspect.Parameter.empty


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
    assert params == ["x", "weight", "bias"], (
        f"_validate_dtypes params {params} must match manifest inputs "
        "['x', 'weight', 'bias'] in order"
    )


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
    weight_other = torch.randn(
        (c,), dtype=dtype, device=torch.device("cuda", 1),
    )
    bias_other = torch.randn(
        (c,), dtype=dtype, device=torch.device("cuda", 1),
    )
    with pytest.raises(ValueError, match="[Dd]evice mismatch"):
        op(x_other, weight_other, bias_other)


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

    weight_same = torch.randn(
        (c,), dtype=dtype, device=torch.device("cuda", 0),
    )
    with pytest.raises(ValueError, match="weight on"):
        op(x, weight_other, bias_same)
    with pytest.raises(ValueError, match="bias on"):
        op(x, weight_same, bias_other)


_OP_CLASSES = [
    pytest.param(InstanceNormFwdOp, "InstanceNormFwdOp", id="InstanceNormFwdOp"),
    pytest.param(
        InstanceNormFwdOpNoAffine,
        "InstanceNormFwdOpNoAffine",
        id="InstanceNormFwdOpNoAffine",
    ),
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
        / "tileops" / "manifest" / "normalization.yaml"
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
def test_instance_norm_affine_rejects_running_stats_path() -> None:
    """The affine variant still defers `use_input_stats=False`."""
    with pytest.raises(NotImplementedError, match="running-stats"):
        InstanceNormFwdOp(
            N=2, C=16, spatial=(8, 8), dtype=torch.float16,
            use_input_stats=False,
        )


@pytest.mark.smoke
def test_instance_norm_no_affine_accepts_running_stats_path() -> None:
    """No-affine variant supports `use_input_stats=False` end-to-end."""
    n, c, spatial, dtype = 2, 16, (8, 8), torch.float16
    op = InstanceNormFwdOpNoAffine(
        N=n, C=c, spatial=spatial, dtype=dtype, use_input_stats=False,
    )
    assert op.use_input_stats is False
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    running_mean = torch.randn(c, dtype=torch.float32, device="cuda")
    running_var = torch.rand(c, dtype=torch.float32, device="cuda") + 0.1
    y = op(x, running_mean, running_var)
    y_ref = F.instance_norm(
        x, running_mean=running_mean, running_var=running_var,
        weight=None, bias=None, use_input_stats=False, eps=1e-5,
    )
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_instance_norm_default_momentum_does_not_change_output() -> None:
    """Per-batch path is independent of `momentum`; default value must match torch."""
    n, c, spatial, dtype = 2, 16, (8, 8), torch.float16
    op_default = InstanceNormFwdOp(N=n, C=c, spatial=spatial, dtype=dtype)
    op_other = InstanceNormFwdOp(
        N=n, C=c, spatial=spatial, dtype=dtype, momentum=0.5,
    )
    assert op_default.momentum == pytest.approx(0.1)
    assert op_other.momentum == pytest.approx(0.5)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    weight = torch.randn((c,), dtype=dtype, device="cuda")
    bias = torch.randn((c,), dtype=dtype, device="cuda")
    y1 = op_default(x, weight, bias)
    y2 = op_other(x, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
