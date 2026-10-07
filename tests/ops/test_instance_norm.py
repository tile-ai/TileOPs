import inspect

import pytest
import torch
import torch.nn.functional as F

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops._signature_codegen import CheckError
from tileops.ops.norm.instance_norm import InstanceNormFwdOp
from workloads.device import run_device
from workloads.norm import InstanceNormWorkload, normalization_verification
from workloads.numerics import compare_outputs, reference_tolerance


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
    test.check(op, *test.gen_inputs())


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
    x_full = torch.randn(shape, dtype=dtype, device=run_device())
    x = x_full[:, :c]  # non-contiguous slice
    weight = torch.randn(c, dtype=dtype, device=run_device())
    bias = torch.randn(c, dtype=dtype, device=run_device())

    op = InstanceNormFwdOp()

    y_ref = F.instance_norm(
        x.contiguous().float(),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)

    y = op(x, weight=weight, bias=bias)
    compare_outputs(y, y_ref, normalization_verification("InstanceNormFwdOp", x.dtype))


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
    x = torch.randn((n, c, *spatial), dtype=dtype, device=run_device())
    y = op(x)
    y_ref = F.instance_norm(
        x.float(),
        weight=None,
        bias=None,
        eps=1e-5,
    ).to(dtype)
    compare_outputs(y, y_ref, normalization_verification("InstanceNormFwdOp", x.dtype))


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
    x = torch.randn((n, c, *spatial), dtype=dtype, device=run_device())
    running_mean = torch.randn(c, dtype=torch.float32, device=run_device())
    running_var = torch.rand(c, dtype=torch.float32, device=run_device()) + 0.1
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
    compare_outputs(y, y_ref, normalization_verification("InstanceNormFwdOp", x.dtype))


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_instance_norm_lazy_cache_reuse_and_respecialization() -> None:
    """One op instance reuses identical specs and caches changed specs."""
    op = InstanceNormFwdOp()

    def run_case(n: int, c: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((n, c, *spatial), dtype=dtype, device=run_device())
        weight = torch.randn((c,), dtype=dtype, device=run_device())
        bias = torch.randn((c,), dtype=dtype, device=run_device())

        y = op(x, weight=weight, bias=bias)
        y_ref = F.instance_norm(
            x.float(),
            weight=weight.float(),
            bias=bias.float(),
            eps=1e-5,
        ).to(dtype)
        compare_outputs(y, y_ref, normalization_verification("InstanceNormFwdOp", x.dtype))

    run_case(2, 8, (4, 4), torch.float16)
    assert op.eval_roofline() == (
        7 * 2 * 8 * 16,
        (2 * 2 * 8 * 16 + 2 * 8) * torch.float16.itemsize,
    )

    run_case(2, 8, (4, 4), torch.float16)

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert op.eval_roofline() == (
        7 * 3 * 12 * 16,
        (2 * 3 * 12 * 16 + 2 * 12) * torch.bfloat16.itemsize,
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_instance_norm_rejects_affine_device_mismatch() -> None:
    """Forward refuses weight or bias on another device than x.

    Without the check the call would dispatch on cross-device tensors, or
    surface as an opaque CUDA error. The device that differs is CPU rather than
    a second GPU: the op compares devices, so one machine with one card
    exercises the same rejection.
    """
    n, c, spatial, dtype = 2, 32, (8, 8), torch.float16
    op = InstanceNormFwdOp()
    x = torch.randn((n, c, *spatial), dtype=dtype, device=run_device())
    elsewhere = torch.randn((c,), dtype=dtype, device="cpu")
    same = torch.randn((c,), dtype=dtype, device=run_device())

    with pytest.raises(CheckError, match="one device"):
        op(x, elsewhere, same)
    with pytest.raises(CheckError, match="one device"):
        op(x, same, elsewhere)


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
    x = torch.randn((n, c, *spatial), dtype=dtype, device=run_device())
    weight = torch.randn((c,), dtype=dtype, device=run_device())
    bias = torch.randn((c,), dtype=dtype, device=run_device())
    y1 = op_default(x, weight=weight, bias=bias)
    y2 = op_other(x, weight=weight, bias=bias)
    assert torch.allclose(y1, y2, **reference_tolerance(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "use_input_stats, affine",
    [(True, "weight"), (True, "bias"), (False, "both"), (True, "none")],
)
def test_instance_norm_matches_torch_on_every_presence_branch(use_input_stats, affine) -> None:
    """Affine tensors are independent, and running statistics are read in eval and updated
    in the input's dtype otherwise, as ``torch.nn.functional.instance_norm`` does."""
    c, dtype = 16, torch.float16
    x = torch.randn((2, c, 8, 8), dtype=dtype, device=run_device()) * 3 + 1
    weight = (
        torch.randn(c, dtype=dtype, device=run_device()) if affine in ("weight", "both") else None
    )
    bias = torch.randn(c, dtype=dtype, device=run_device()) if affine in ("bias", "both") else None
    stats = (torch.randn(c, device=run_device()), torch.rand(c, device=run_device()) + 0.5)
    mine, ref = [s.clone() for s in stats], [s.clone() for s in stats]
    op = InstanceNormFwdOp(use_input_stats=use_input_stats)
    y = op(x, *mine, weight, bias)
    y_ref = F.instance_norm(x, *ref, weight, bias, use_input_stats=use_input_stats)
    compare_outputs(y, y_ref, normalization_verification("InstanceNormFwdOp", x.dtype))
    compare_outputs(mine, ref, normalization_verification("InstanceNormFwdOp", x.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("n, spatial", [(20, (33,)), (3, (5000,))])
def test_instance_norm_updates_running_statistics_across_blocks(n, spatial) -> None:
    """Several blocks per channel update the running statistics as torch does, for a
    register-held row and a shared-memory-staged one."""
    c, dtype = 3, torch.float16
    x = torch.randn((n, c, *spatial), dtype=dtype, device=run_device()) * 3 + 1
    stats = (torch.randn(c, device=run_device()), torch.rand(c, device=run_device()) + 0.5)
    mine, ref = [s.clone() for s in stats], [s.clone() for s in stats]
    y = InstanceNormFwdOp(momentum=0.3)(x, *mine)
    y_ref = F.instance_norm(x, *ref, momentum=0.3)
    compare_outputs(y, y_ref, normalization_verification("InstanceNormFwdOp", x.dtype))
    compare_outputs(mine, ref, normalization_verification("InstanceNormFwdOp", x.dtype))


@pytest.mark.smoke
def test_instance_norm_needs_both_running_statistics_to_read_them() -> None:
    x = torch.randn((2, 16, 8, 8), dtype=torch.float16, device=run_device())
    stat = torch.zeros(16, device=run_device())
    with pytest.raises(ValueError, match="use_input_stats or present"):
        InstanceNormFwdOp(use_input_stats=False)(x)
    with pytest.raises(ValueError, match="'running_var' is required"):
        InstanceNormFwdOp()(x, stat)


def _misaligned(t: torch.Tensor) -> torch.Tensor:
    """*t* copied into a contiguous view that starts one element into its storage."""
    view = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:].view(t.shape)
    view.copy_(t)
    return view


@pytest.mark.smoke
@pytest.mark.parametrize("stats", [False, True], ids=["affine", "running-stats"])
def test_instance_norm_reads_input_off_the_vector_boundary(stats: bool) -> None:
    """A contiguous input starting one element into its storage matches torch, whether the
    call normalizes with an affine or updates running statistics."""
    x = _misaligned(torch.randn(2, 16, 16, 16, dtype=torch.float16, device=run_device()))
    weight = None if stats else torch.randn(16, dtype=x.dtype, device=x.device)
    bias = None if stats else torch.randn(16, dtype=x.dtype, device=x.device)
    rm = torch.zeros(16, device=x.device) if stats else None
    rv = torch.ones(16, device=x.device) if stats else None
    y_ref = F.instance_norm(
        x.float(),
        None if rm is None else rm.clone(),
        None if rv is None else rv.clone(),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        use_input_stats=True,
    ).to(x.dtype)
    y = InstanceNormFwdOp()(x, rm, rv, weight, bias)
    compare_outputs(y, y_ref, normalization_verification("InstanceNormFwdOp", x.dtype))
