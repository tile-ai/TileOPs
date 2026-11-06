import math

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.norm import GroupNormSplitKernel
from tileops._ops._signature_codegen import CheckError
from tileops._ops.norm.group_norm import GroupNormFwdOp
from workloads.device import run_device
from workloads.norm import (
    GroupNormFarFromZeroWorkload,
    GroupNormOutlierWorkload,
    GroupNormWorkload,
)


class GroupNormTest(GroupNormWorkload, TestBase):
    pass


class GroupNormFarFromZeroTest(GroupNormFarFromZeroWorkload, TestBase):
    pass


class GroupNormOutlierTest(GroupNormOutlierWorkload, TestBase):
    pass


_FAR_INPUTS = [
    (torch.float32, "constant"),
    (torch.float32, "noisy"),
    (torch.float32, "huge"),
    (torch.bfloat16, "huge"),
]
# The other kernel of a path (with or without the affine) shares its statistics: it takes
# the rows that reach its own write.
_FAR_WRITE_INPUTS = [(torch.float32, "constant"), (torch.float32, "huge")]


def _path(kernel) -> str:
    """The body a built GroupNorm kernel runs."""
    if isinstance(kernel, GroupNormSplitKernel):
        return "split"
    if kernel._warp_row_vector() is not None:
        return "warp"
    if kernel._held_row_vector() is not None:
        return "held"
    return "staged"


def _far_group_case(shape, path, affine, dtype, row, mark):
    n, c, spatial, g = shape
    width = c // g * math.prod(spatial)
    tag = f"{path}-{width}-{'affine' if affine else 'no-affine'}-{str(dtype)[6:]}-{row}"
    return pytest.param(n, c, spatial, g, path, affine, dtype, row, marks=mark, id=tag)


# A padded row staged in shared memory (4003 of 4096 columns), a row the registers hold
# (32768), a wider row the access divides held a block by vectors (6272), and rows cut
# across blocks (60000 columns, and one row of 80001).
_GROUP_NORM_FAR_CASES = [
    _far_group_case(shape, path, affine, dtype, row, mark)
    for shape, path, affine, mark, inputs in [
        ((2, 1, (4003,), 1), "staged", True, pytest.mark.smoke, _FAR_INPUTS),
        ((2, 32, (64, 64), 4), "held", False, pytest.mark.smoke, _FAR_INPUTS),
        ((2, 8, (784,), 1), "held", True, pytest.mark.smoke, _FAR_INPUTS),
        ((2, 1, (4003,), 1), "staged", False, pytest.mark.full, _FAR_WRITE_INPUTS),
        ((2, 32, (64, 64), 4), "held", True, pytest.mark.full, _FAR_WRITE_INPUTS),
        ((2, 8, (784,), 1), "held", False, pytest.mark.full, _FAR_WRITE_INPUTS),
        ((1, 1, (80001,), 1), "split", True, pytest.mark.full, _FAR_INPUTS),
        ((1, 1, (80001,), 1), "split", False, pytest.mark.full, _FAR_WRITE_INPUTS),
        ((2, 128, (100, 150), 32), "split", True, pytest.mark.full, _FAR_INPUTS),
        ((2, 128, (100, 150), 32), "split", False, pytest.mark.full, _FAR_WRITE_INPUTS),
    ]
    for dtype, row in inputs
] + [
    # Constant 1e36 rows whose references a sum of raw values would round off the row: a
    # held row of 12288 (12 reference elements a thread) and 512 split pieces of 4096.
    _far_group_case(shape, path, True, torch.float32, "huge", pytest.mark.full)
    for shape, path in [((1, 1, (12288,), 1), "held"), ((1, 1, (2097152,), 1), "split")]
]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("n, c, spatial, g, path, affine, dtype, row", _GROUP_NORM_FAR_CASES)
def test_group_norm_normalizes_groups_far_from_zero(
    n, c, spatial, g, path, affine, dtype, row
) -> None:
    test = GroupNormFarFromZeroTest(n, c, spatial, g, dtype, row=row)
    x, weight, bias = test.gen_inputs()
    op = GroupNormFwdOp(num_groups=g)
    test.check(op, *((x, weight, bias) if affine else (x,)))
    (built,) = op.iter_kernels()
    assert _path(built) == path


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "n, c, spatial, g, stride, affine, path",
    [
        # One warp a group of 1024; one block a group of 4096 and of 40000
        pytest.param(2, 32, (32, 32), 32, None, True, "warp", marks=pytest.mark.smoke),
        pytest.param(2, 32, (64, 64), 32, None, False, "held", marks=pytest.mark.smoke),
        pytest.param(1, 1, (40000,), 1, None, True, "held", marks=pytest.mark.smoke),
        # One block a group: 1999 by single elements, and 32768
        pytest.param(2, 2, (1999,), 2, None, True, "held", marks=pytest.mark.full),
        pytest.param(2, 32, (64, 64), 4, None, False, "held", marks=pytest.mark.full),
        # A padded row staged in shared memory
        pytest.param(1, 1, (12001,), 1, None, True, "staged", marks=pytest.mark.full),
        # Rows cut into pieces of 4096: the far element leads the row, then every piece
        pytest.param(2, 128, (100, 150), 32, None, False, "split", marks=pytest.mark.full),
        pytest.param(2, 128, (100, 150), 32, 1024, True, "split", marks=pytest.mark.full),
    ],
)
def test_group_norm_normalizes_groups_led_by_an_outlier(
    n, c, spatial, g, stride, affine, path
) -> None:
    test = GroupNormOutlierTest(n, c, spatial, g, torch.float32, stride=stride)
    x, weight, bias = test.gen_inputs()
    op = GroupNormFwdOp(num_groups=g)
    test.check(op, *((x, weight, bias) if affine else (x,)))
    (built,) = op.iter_kernels()
    assert _path(built) == path


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_group_norm_squares_far_differences_within_range() -> None:
    """A lane's four elements of 5e18 among zeros: their differences from its reference square
    past fp32's range unless scaled, where the group's square sum about its mean (about
    1e38) does not."""
    x = torch.zeros(1, 1, 1024, device=run_device())
    x[..., :4] = 5e18
    weight, bias = torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
    op = GroupNormFwdOp(num_groups=1)
    y = op(x, weight, bias)
    (built,) = op.iter_kernels()
    assert _path(built) == "warp"
    expected = torch.nn.functional.group_norm(x.double(), 1, weight.double(), bias.double())
    torch.testing.assert_close(y, expected.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "width, path",
    [
        pytest.param(1024, "warp", marks=pytest.mark.smoke),
        pytest.param(4096, "held", marks=pytest.mark.full),
    ],
)
def test_group_norm_takes_weights_near_fp32_range(width, path) -> None:
    """Rows of alternating -1 and +1 scaled by a weight of 5e37: the outputs are about
    +-5e37, finite. The write folds rstd, the inverse of the differences'
    exact power-of-two scale (8 here) and the weight into one coefficient only where that
    coefficient stays in fp32's range."""
    x = torch.ones(1, 1, width, device=run_device())
    x[..., ::2] = -1.0
    weight = torch.full((1,), 5e37, device=x.device)
    bias = torch.zeros(1, device=x.device)
    op = GroupNormFwdOp(num_groups=1)
    y = op(x, weight, bias)
    (built,) = op.iter_kernels()
    assert _path(built) == path
    expected = torch.nn.functional.group_norm(x.double(), 1, weight.double(), bias.double())
    torch.testing.assert_close(y, expected.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_group_norm_merges_far_lane_means_within_range() -> None:
    """Four elements of 4e18 in lane 0 of a 128-wide row, zeros elsewhere: the lanes' square
    sum about lane 0's mean is 31 times 4e18 squared, past fp32's range, where the row's
    square sum about its mean (about 6.2e37) is not."""
    x = torch.zeros(1, 1, 128, device=run_device())
    x[..., :4] = 4e18
    weight, bias = torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
    op = GroupNormFwdOp(num_groups=1)
    y = op(x, weight, bias)
    (built,) = op.iter_kernels()
    assert _path(built) == "warp"
    expected = torch.nn.functional.group_norm(x.double(), 1, weight.double(), bias.double())
    torch.testing.assert_close(y, expected.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "width, path, size, eps_exp",
    [
        pytest.param(1024, "warp", 1.3e-19, -126, marks=pytest.mark.smoke),
        pytest.param(1024, "warp", 1.3e-20, -132, marks=pytest.mark.smoke),
        pytest.param(1024, "warp", 1.3e-21, -138, marks=pytest.mark.full),
        pytest.param(4096, "held", 1.3e-19, -126, marks=pytest.mark.full),
        pytest.param(4096, "held", 1.3e-20, -132, marks=pytest.mark.full),
        pytest.param(4096, "held", 1.3e-21, -138, marks=pytest.mark.full),
    ],
)
def test_group_norm_keeps_small_squares_normal(width, path, size, eps_exp) -> None:
    """Rows of alternating -size and +size with eps = 2^eps_exp: each square sits near or
    below fp32's smallest normal, where unscaled squares still hold the result. A warp whose
    scaled square sums are that small takes its statistics again unscaled, so the result
    matches float64."""
    x = torch.full((1, 1, width), size, device=run_device())
    x[..., ::2] = -size
    weight, bias = torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
    op = GroupNormFwdOp(num_groups=1, eps=2.0**eps_exp)
    y = op(x, weight, bias)
    (built,) = op.iter_kernels()
    assert _path(built) == path
    expected = torch.nn.functional.group_norm(
        x.double(), 1, weight.double(), bias.double(), eps=2.0**eps_exp
    )
    torch.testing.assert_close(y, expected.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "width, path, exponent",
    [
        pytest.param(1024, "warp", 55, marks=pytest.mark.smoke),
        pytest.param(1024, "warp", 56, marks=pytest.mark.smoke),
        pytest.param(1024, "warp", 54, marks=pytest.mark.full),
        pytest.param(4096, "held", 53, marks=pytest.mark.full),
        pytest.param(4096, "held", 54, marks=pytest.mark.full),
        pytest.param(4096, "held", 55, marks=pytest.mark.full),
    ],
)
def test_group_norm_scales_around_the_range_bound(width, path, exponent) -> None:
    """Lane 0's 4-element vectors at 2^exponent, the rest of the row at -2^exponent: every
    lane's square sum about its reference is zero, so the references alone decide, and the
    powers span the bound on them past which a warp keeps its scale (2^55 for 1024
    elements, 2^54 for 4096). Unscaled at the bound, the lanes' square sum about lane 0's
    mean reaches 2^-3 of fp32's range."""
    x = torch.full((1, 1, width), -(2.0**exponent), device=run_device())
    lane_zero = (torch.arange(width, device=x.device) // 4) % 32 == 0
    x[..., lane_zero] = 2.0**exponent
    weight, bias = torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
    op = GroupNormFwdOp(num_groups=1)
    y = op(x, weight, bias)
    (built,) = op.iter_kernels()
    assert _path(built) == path
    expected = torch.nn.functional.group_norm(x.double(), 1, weight.double(), bias.double())
    torch.testing.assert_close(y, expected.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_group_norm_scales_a_warp_of_one_element_threads() -> None:
    """A 3-element row puts one element in each thread, whose own square sum needs no scale
    while the warp's merge still takes the lanes' factor."""
    x = torch.tensor([[[-1.0, 0.0, 1.0]]], device=run_device())
    weight, bias = torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
    op = GroupNormFwdOp(num_groups=1)
    y = op(x, weight, bias)
    (built,) = op.iter_kernels()
    assert _path(built) == "held"
    expected = torch.nn.functional.group_norm(x.double(), 1, weight.double(), bias.double())
    torch.testing.assert_close(y, expected.float(), rtol=1e-5, atol=1e-5)


class GroupNormFixture(FixtureBase):
    PARAMS = [
        (
            "n, c, spatial, g, dtype, tune",
            [
                # Small CI-friendly shapes -- fp32
                pytest.param(2, 32, (8, 8), 8, torch.float32, False, marks=pytest.mark.smoke),
                # Small CI-friendly shapes -- fp16
                pytest.param(2, 32, (8, 8), 8, torch.float16, False, marks=pytest.mark.smoke),
                # Small CI-friendly shapes -- bf16
                pytest.param(2, 32, (8, 8), 8, torch.bfloat16, False, marks=pytest.mark.smoke),
                # Wide groups: Stable Diffusion's 64 x 64 latent (40960 a group, held in
                # registers), an SDXL latent no block holds, a bf16 group of 262144 cut
                # across blocks, and an fp32 row past SM89's shared memory.
                pytest.param(2, 320, (64, 64), 32, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(2, 320, (152, 104), 32, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1, 128, (256, 256), 32, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 128, (50, 125), 32, torch.float32, False, marks=pytest.mark.full),
                # A padded FP32 row also exceeds SM90 shared memory.
                pytest.param(2, 128, (100, 150), 32, torch.float32, False, marks=pytest.mark.full),
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
            ],
        ),
    ]


@GroupNormFixture
def test_group_norm_op(
    n: int, c: int, spatial: tuple, g: int, dtype: torch.dtype, tune: bool
) -> None:
    test = GroupNormTest(n, c, spatial, g, dtype)
    op = GroupNormFwdOp(num_groups=g)
    test.check(op, *test.gen_inputs())


class GroupNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "n, c, spatial, g, dtype",
            [
                pytest.param(2, 32, (8, 8), 8, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 32, (8, 8), 8, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@GroupNormNonContigFixture
def test_group_norm_non_contiguous(
    n: int, c: int, spatial: tuple, g: int, dtype: torch.dtype
) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    test = GroupNormTest(n, c, spatial, g, dtype)
    _, weight, bias = test.gen_inputs()
    x = torch.randn((n, c * 2, *spatial), dtype=dtype, device=run_device())[:, :c]
    test.check(GroupNormFwdOp(num_groups=g), x, weight, bias)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_group_norm_lazy_cache_reuse_and_respecialization() -> None:
    """One op instance reuses identical specs and caches changed specs."""
    op = GroupNormFwdOp(num_groups=4)

    def run_case(n: int, c: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        test = GroupNormTest(n, c, spatial, 4, dtype)
        test.check(op, *test.gen_inputs())

    run_case(2, 16, (4, 4), torch.float16)
    assert op.eval_roofline() == (
        7 * 2 * 16 * 16,
        (2 * 2 * 16 * 16 + 2 * 16) * torch.float16.itemsize,
    )

    run_case(2, 16, (4, 4), torch.float16)

    run_case(3, 24, (2, 8), torch.bfloat16)
    assert op.eval_roofline() == (
        7 * 3 * 24 * 16,
        (2 * 3 * 24 * 16 + 2 * 24) * torch.bfloat16.itemsize,
    )


@pytest.mark.skipif(
    torch.device(run_device()).type == "cpu",
    reason="mixes a run-device tensor with a cpu one, so the run device is not cpu",
)
@pytest.mark.smoke
def test_group_norm_rejects_affine_device_mismatch() -> None:
    """Forward refuses weight or bias on another device than x.

    Without the check the call would dispatch on cross-device tensors, or
    surface as an opaque CUDA error. The device that differs is CPU rather than
    a second GPU: the op compares devices, so one machine with one card
    exercises the same rejection.
    """
    n, c, spatial, g, dtype = 2, 32, (8, 8), 8, torch.float16
    op = GroupNormFwdOp(num_groups=g)
    x = torch.randn((n, c, *spatial), dtype=dtype, device=run_device())
    elsewhere = torch.randn((c,), dtype=dtype, device="cpu")
    same = torch.randn((c,), dtype=dtype, device=run_device())

    with pytest.raises(CheckError, match="one device"):
        op(x, elsewhere, same)
    with pytest.raises(CheckError, match="one device"):
        op(x, same, elsewhere)


class GroupNormNoAffineFixture(FixtureBase):
    PARAMS = [
        (
            "n, c, spatial, g, dtype",
            [
                pytest.param(2, 32, (8, 8), 8, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 32, (8, 8), 8, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 32, (8, 8), 8, torch.bfloat16, marks=pytest.mark.smoke),
                # Wide groups, cut across blocks and held in registers.
                pytest.param(2, 320, (152, 104), 32, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 320, (64, 64), 32, torch.float16, marks=pytest.mark.full),
                pytest.param(4, 16, (4, 4), 4, torch.float16, marks=pytest.mark.full),
                # Non-aligned spatial: exercises padding path.
                pytest.param(2, 32, (7, 7), 8, torch.float16, marks=pytest.mark.full),
                # 1D spatial.
                pytest.param(2, 32, (16,), 8, torch.float16, marks=pytest.mark.full),
                # 3D spatial.
                pytest.param(2, 16, (4, 4, 4), 4, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@GroupNormNoAffineFixture
def test_group_norm_no_affine_op(
    n: int, c: int, spatial: tuple, g: int, dtype: torch.dtype
) -> None:
    """No-affine GroupNorm op matches torch.nn.functional.group_norm with weight=bias=None."""
    test = GroupNormTest(n, c, spatial, g, dtype)
    x, _, _ = test.gen_inputs()
    test.check(GroupNormFwdOp(num_groups=g), x)


@pytest.mark.smoke
def test_group_norm_forward_signature() -> None:
    """One forward takes input plus the optional affine pair."""
    import inspect

    sig = inspect.signature(GroupNormFwdOp.forward)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["input", "weight", "bias"], f"got {params}"
    for name in ("weight", "bias"):
        assert sig.parameters[name].default is None, f"{name} must default to None"


@pytest.mark.smoke
@pytest.mark.parametrize("give", ["weight", "bias"])
def test_group_norm_takes_either_affine_tensor_alone(give: str) -> None:
    """weight and bias are independent, as in ``torch.nn.functional.group_norm``."""
    test = GroupNormTest(2, 32, (8, 8), 8, torch.float16)
    x, weight, bias = test.gen_inputs()
    inputs = (x, weight) if give == "weight" else (x, None, bias)
    test.check(GroupNormFwdOp(num_groups=8), *inputs)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, c, spatial, g",
    [
        # Very few rows, so the grid is one or two blocks wide.
        (1, 24, (4, 4), 3),  # M = 3
        (3, 30, (2, 2), 5),  # M = 15
        (1, 16, (8, 8), 1),  # M = 1
    ],
)
def test_group_norm_no_affine_tail_block(n: int, c: int, spatial: tuple, g: int) -> None:
    """No-affine GroupNorm handles a row count smaller than one grid block."""
    test = GroupNormTest(n, c, spatial, g, torch.float16)
    x, _, _ = test.gen_inputs()
    test.check(GroupNormFwdOp(num_groups=g), x)


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_group_norm_under_tuning(tune: bool) -> None:
    """A group of 8192 elements, past the register-held row, where multi-row blocks are
    candidates."""
    test = GroupNormTest(2, 32, (32, 32), 4, torch.float16)
    op = GroupNormFwdOp(num_groups=4)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_group_norm_constant_group() -> None:
    """A group of equal elements normalizes to zero, at a width the row is cut across blocks."""
    test = GroupNormTest(1, 32, (4096,), 1, torch.float16)
    x, _, _ = test.gen_inputs()
    test.check(GroupNormFwdOp(num_groups=1), torch.full_like(x, 10000.0))


def _misaligned(t: torch.Tensor) -> torch.Tensor:
    """*t* copied into a contiguous view that starts one element into its storage."""
    view = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:].view(t.shape)
    view.copy_(t)
    return view


@pytest.mark.smoke
@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no-affine"])
def test_group_norm_reads_inputs_off_the_vector_boundary(affine: bool) -> None:
    """A contiguous input, weight and bias may start anywhere in their storage."""
    test = GroupNormTest(2, 32, (16, 16), 8, torch.float16)
    x, weight, bias = (_misaligned(t) for t in test.gen_inputs())
    test.check(GroupNormFwdOp(num_groups=8), *((x, weight, bias) if affine else (x,)))
