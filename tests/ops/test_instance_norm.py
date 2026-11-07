import inspect

import pytest
import torch
import torch.nn.functional as F

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.norm import InstanceNormFwdTrainKernel, InstanceNormFwdTrainSingleKernel
from tileops._ops._signature_codegen import CheckError
from tileops._ops.norm.instance_norm import InstanceNormFwdOp
from workloads.device import run_device
from workloads.norm import (
    InstanceNormFarFromZeroWorkload,
    InstanceNormOutlierWorkload,
    InstanceNormWorkload,
    _norm_reference,
    assert_mean_close,
    far_from_zero,
    normalization_verification,
)
from workloads.numerics import compare_outputs, reference_tolerance


class InstanceNormTest(InstanceNormWorkload, TestBase):
    pass


class InstanceNormFarFromZeroTest(InstanceNormFarFromZeroWorkload, TestBase):
    pass


class InstanceNormOutlierTest(InstanceNormOutlierWorkload, TestBase):
    pass


@pytest.mark.parametrize(
    "spatial, dtype, row",
    [
        # A padded row staged in shared memory
        pytest.param((4003,), torch.float32, "noisy", marks=pytest.mark.smoke),
        pytest.param((4003,), torch.float32, "constant", marks=pytest.mark.smoke),
        pytest.param((4003,), torch.float32, "huge", marks=pytest.mark.smoke),
        pytest.param((4003,), torch.bfloat16, "huge", marks=pytest.mark.smoke),
        # A wide row held in registers
        pytest.param((40960,), torch.float32, "noisy", marks=pytest.mark.full),
        pytest.param((40960,), torch.bfloat16, "huge", marks=pytest.mark.full),
    ],
)
def test_instance_norm_normalizes_rows_far_from_zero(spatial, dtype, row) -> None:
    test = InstanceNormFarFromZeroTest(2, 3, spatial, dtype, row=row)
    test.check(InstanceNormFwdOp(), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "n, spatial, kernel, path",
    [
        # One block a channel: one warp a sample, register fragments, and a padded row
        # staged in shared memory
        pytest.param(2, (1024,), InstanceNormFwdTrainSingleKernel, "warp", marks=pytest.mark.smoke),
        pytest.param(
            2, (64, 64), InstanceNormFwdTrainSingleKernel, "registers", marks=pytest.mark.smoke
        ),
        pytest.param(2, (4003,), InstanceNormFwdTrainKernel, "staged", marks=pytest.mark.smoke),
        # Blocks over the batch: one warp a sample, and register fragments; a row of 40960
        pytest.param(300, (1024,), InstanceNormFwdTrainKernel, "warp", marks=pytest.mark.full),
        # One warp a sample of 784: the first four lanes hold a vector more than the rest
        pytest.param(2, (784,), InstanceNormFwdTrainSingleKernel, "warp", marks=pytest.mark.full),
        pytest.param(16, (64, 64), InstanceNormFwdTrainKernel, "registers", marks=pytest.mark.full),
        pytest.param(2, (40960,), InstanceNormFwdTrainKernel, "registers", marks=pytest.mark.full),
    ],
)
def test_instance_norm_trains_on_rows_led_by_an_outlier(n, spatial, kernel, path) -> None:
    test = InstanceNormOutlierTest(n, 3, spatial, torch.float32)
    x, mean, var, weight, bias = test.gen_inputs()
    op = InstanceNormFwdOp()
    test.check(op, x, mean.clone(), var.clone(), weight, bias)
    (built,) = op.iter_kernels()
    if built._warp_row_vector() is not None:
        body = "warp"
    elif built._holds_row_in_registers(built.D, built.D_padded):
        body = "registers"
    else:
        body = "staged"
    assert (type(built), body) == (kernel, path)
    stats = [mean.clone(), var.clone()]
    op(x, *stats, weight, bias)
    expected = [mean.double(), var.double()]
    F.instance_norm(x.double(), *expected, weight.double(), bias.double(), use_input_stats=True)
    compare_outputs(
        stats,
        [s.float() for s in expected],
        normalization_verification("InstanceNormFwdOp", x.dtype),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "lead, rest, eps",
    [
        pytest.param(1.3e-19, -1.3e-19, 2.0**-126, marks=pytest.mark.smoke),
        pytest.param(1.3e-20, -1.3e-20, 2.0**-132, marks=pytest.mark.smoke),
        pytest.param(4e18, 0.0, 1e-5, marks=pytest.mark.smoke),
        pytest.param(1.3e-21, -1.3e-21, 2.0**-138, marks=pytest.mark.full),
        pytest.param(2.0**55, -(2.0**55), 1e-5, marks=pytest.mark.full),
        pytest.param(2.0**56, -(2.0**56), 1e-5, marks=pytest.mark.full),
    ],
)
def test_instance_norm_warp_rows_keep_their_squares_in_range(lead, rest, eps) -> None:
    """Warp rows of 1024: alternating signs for the small sizes, lane 0's 4-element vectors
    at *lead* ahead of *rest* otherwise. Squares near fp32's smallest normal stay unscaled;
    lanes whose references reach the range bound (2^55 here) or whose squares are large keep
    the scale, so the lanes' square sums about lane 0's mean stay finite."""
    x = torch.full((300, 1, 1024), rest, device=run_device())
    if abs(lead) < 1.0:
        x[..., ::2] = lead
    else:
        x[..., (torch.arange(1024, device=x.device) // 4) % 32 == 0] = lead
    weight, bias = torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
    op = InstanceNormFwdOp(eps=eps)
    y = op(x, None, None, weight, bias)
    (built,) = op.iter_kernels()
    assert built._warp_row_vector() is not None
    expected = F.instance_norm(x.double(), weight=weight.double(), bias=bias.double(), eps=eps)
    torch.testing.assert_close(y, expected.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "spatial",
    [pytest.param((16,), marks=pytest.mark.smoke), pytest.param((4003,), marks=pytest.mark.full)],
)
def test_instance_norm_running_mean_of_rows_far_from_zero(spatial) -> None:
    # The running mean averages a batch of updates near 1e35, whose sum is past fp32's range.
    x = far_from_zero((4096, 1, *spatial), "huge")
    mean = torch.zeros(1, dtype=torch.float32, device=x.device)
    var = torch.ones_like(mean)
    InstanceNormFwdOp()(x, mean, var)
    # The reference takes x less its first element in float64, then puts that element back
    # into the mean: torch's float64 statistics of a 1e36 row are off by more than they hold.
    lead = x.flatten()[0].double()
    expected_mean, expected_var = torch.zeros_like(mean).double(), torch.ones_like(var).double()
    F.instance_norm(x.double() - lead, expected_mean, expected_var, use_input_stats=True)
    expected_mean += 0.1 * lead
    # A 4096-term fp32 sum rounds by up to 4096 ulps of the mean. The running mean takes
    # momentum's share of the rows' means, so its deviation is that share of theirs.
    tolerance = dict(rtol=4096 * torch.finfo(torch.float32).eps, atol=0.0)
    deviation = 0.1 * (x.double() - lead).mean(dim=(1, 2)).std(correction=0)
    assert_mean_close(mean, expected_mean, deviation, tolerance=1e-5, adds=4096)
    torch.testing.assert_close(var, expected_var.float(), **tolerance)


@pytest.mark.parametrize(
    "n, spatial, dtype",
    [
        # One warp a sample, one block a channel writing the running statistics itself
        pytest.param(2, (16,), torch.float32, marks=pytest.mark.smoke),
        # 16-bit updates of a power-of-two batch, divided by it before the sum: one block a
        # channel, and a padded row staged in shared memory
        pytest.param(8, (16,), torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param(4, (4003,), torch.float16, marks=pytest.mark.smoke),
        # One warp a sample, blocks over the batch leaving partial sums to a second kernel
        pytest.param(300, (16,), torch.float32, marks=pytest.mark.full),
        # A padded row staged in shared memory
        pytest.param(2, (4003,), torch.float32, marks=pytest.mark.full),
        # A bf16 batch whose quotient of the smallest bf16 value would fall below fp32's
        pytest.param(2**17, (16,), torch.bfloat16, marks=pytest.mark.full),
    ],
)
def test_instance_norm_running_statistics_keep_subnormal_values(n, spatial, dtype) -> None:
    # With momentum 0 every sample's update is the old statistic, so their batch mean is it:
    # the smallest subnormal the running statistics hold, rounded to the input's dtype.
    x = torch.randn(n, 1, *spatial, dtype=dtype, device=run_device())
    smallest = {torch.float32: 2.0**-149, torch.bfloat16: 2.0**-133, torch.float16: 2.0**-24}
    tiny = torch.full((1,), smallest[dtype], device=x.device)
    mean, var = tiny.clone(), tiny.clone()
    InstanceNormFwdOp(momentum=0.0)(x, mean, var)
    torch.testing.assert_close(mean, tiny, rtol=0, atol=0)
    torch.testing.assert_close(var, tiny, rtol=0, atol=0)


@pytest.mark.parametrize(
    "n, spatial, case",
    [
        # Seven updates of FMAX: their sum is past fp32's range, their mean is FMAX
        pytest.param(7, (16,), "fmax", marks=pytest.mark.smoke),
        # Eight updates of +2^127, then eight of -2^127 in another block: the blocks' sums
        # are +inf and -inf, the mean is 0
        pytest.param(16, (16,), "opposite", marks=pytest.mark.full),
        # Rows alternating +-1e20: each variance update is inf, and so is their mean variance
        pytest.param(2, (4003,), "infinite", marks=pytest.mark.full),
    ],
)
def test_instance_norm_running_statistics_past_fp32_sum_range(n, spatial, case) -> None:
    fmax = torch.finfo(torch.float32).max
    mean = torch.zeros(1, device=run_device())
    var = torch.ones(1, device=run_device())
    if case == "fmax":
        x = torch.randn(n, 1, *spatial, device=run_device())
        momentum, expected_mean, expected_var = 0.0, fmax, fmax
        mean.fill_(fmax)
        var.fill_(fmax)
    elif case == "opposite":
        x = torch.full((n, 1, *spatial), 2.0**127, device=run_device())
        x[n // 2 :] = -(2.0**127)
        momentum, expected_mean, expected_var = 1.0, 0.0, 0.0
    else:
        x = torch.full((n, 1, *spatial), 1e20, device=run_device())
        x[..., 1::2] = -1e20
        momentum, expected_mean, expected_var = 1.0, None, float("inf")
    InstanceNormFwdOp(momentum=momentum)(x, mean, var)
    if expected_mean is not None:
        torch.testing.assert_close(mean, torch.full_like(mean, expected_mean), rtol=0, atol=0)
    torch.testing.assert_close(var, torch.full_like(var, expected_var), rtol=0, atol=0)


@pytest.mark.smoke
def test_instance_norm_launches_batches_past_one_grid_dimension() -> None:
    # 600000 samples of one channel take 75000 blocks, past the 65535 a second grid
    # dimension holds.
    x = torch.randn(600000, 1, 16, device=run_device())
    mean, var = torch.zeros(1, device=x.device), torch.ones(1, device=x.device)
    ref_mean, ref_var = mean.clone(), var.clone()
    y = InstanceNormFwdOp()(x, mean, var)
    y_ref = F.instance_norm(x, ref_mean, ref_var, use_input_stats=True)
    compare_outputs(y, y_ref, normalization_verification("InstanceNormFwdOp", x.dtype))
    torch.testing.assert_close(mean, ref_mean)
    torch.testing.assert_close(var, ref_var)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_instance_norm_reference_reads_running_statistics_in_input_dtype(dtype) -> None:
    # The manifest's oracle computes in fp32; torch reads the statistics rounded to the input.
    x = torch.randn(2, 16, 8, 8, dtype=dtype, device=run_device())
    weight, bias = (torch.randn(16, dtype=dtype, device=x.device) for _ in range(2))
    stats = (3 * torch.randn(16, device=x.device), torch.rand(16, device=x.device) + 0.5)
    inputs = (x, *(s.clone() for s in stats), weight, bias)
    params = dict(eps=1e-5, use_input_stats=False, momentum=0.1)
    expected = F.instance_norm(x, *(s.clone() for s in stats), weight, bias, False, 0.1, 1e-5)
    assert torch.equal(_norm_reference("InstanceNormFwdOp", inputs, params), expected)


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
                # The first float32 row of whole vectors past what one warp holds
                pytest.param(4, 8, (4, 257), torch.float32, False, marks=pytest.mark.full),
                # The widest 16-bit row one warp holds
                pytest.param(4, 8, (32, 32), torch.float16, False, marks=pytest.mark.full),
                # Rows one warp holds, eight a block, the last block past the batch
                pytest.param(4, 255, (8, 8), torch.bfloat16, False, marks=pytest.mark.full),
                # 1D spatial, the first 16-bit row of whole vectors past what one warp holds
                pytest.param(2, 16, (1032,), torch.float16, False, marks=pytest.mark.full),
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


@pytest.mark.skipif(
    torch.device(run_device()).type == "cpu",
    reason="mixes a run-device tensor with a cpu one, so the run device is not cpu",
)
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
@pytest.mark.parametrize(
    "n, spatial, dtype",
    [
        (20, (33,), torch.float16),
        (20, (28, 28), torch.float16),
        (20, (32, 32), torch.float32),
        (3, (5000,), torch.float16),
        (1, (192, 192), torch.float16),
        (3, (192, 192), torch.float16),
        (1, (192, 192), torch.bfloat16),
        (3, (192, 192), torch.bfloat16),
        (1, (192, 192), torch.float32),
        (3, (192, 192), torch.float32),
    ],
)
def test_instance_norm_updates_running_statistics_across_blocks(n, spatial, dtype) -> None:
    """Single and multiple blocks preserve output and running statistics, including
    wide register rows whose reduction must avoid scalar shift loads, and rows one warp
    holds whose last vectors and last block run past the row and the batch, up to the
    widest float32 row one warp holds."""
    c = 3
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
