import math

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase, smoke_first
from tileops._kernels.norm import GroupNormSplitKernel
from tileops._ops._signature_codegen import CheckError
from tileops._ops.norm.group_norm import GroupNormFwdOp
from workloads.device import run_device
from workloads.norm import (
    GroupNormFarFromZeroWorkload,
    GroupNormOutlierWorkload,
    GroupNormWorkload,
    normalization_verification,
)
from workloads.numerics import compare_outputs


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
    # A smoke shape keeps the fp32 and bf16 rows of one kind; the other rows run in full.
    if mark.name == "smoke" and dtype == torch.float32 and row != "huge":
        mark = pytest.mark.full
    n, c, spatial, g = shape
    width = c // g * math.prod(spatial)
    tag = f"{path}-{width}-{'affine' if affine else 'no-affine'}-{str(dtype)[6:]}-{row}"
    return pytest.param(n, c, spatial, g, path, affine, dtype, row, marks=mark, id=tag)


# A padded row staged in shared memory (4003 of 4096 columns), a row the registers hold
# (32768), a wider row the access divides held a block by vectors (6272), and rows cut
# across blocks (60000 columns, and one row of 80001).
_GROUP_NORM_FAR_CASES = smoke_first(
    [
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
    ]
    + [
        # Constant 1e36 rows whose references a sum of raw values would round off the row: a
        # held row of 12288 (12 reference elements a thread) and 512 split pieces of 4096.
        _far_group_case(shape, path, True, torch.float32, "huge", pytest.mark.full)
        for shape, path in [((1, 1, (12288,), 1), "held"), ((1, 1, (2097152,), 1), "split")]
    ],
)


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


def _extreme_row(kind: str, width: int, value: float) -> torch.Tensor:
    """A ``(1, 1, width)`` row: zeros led by four ``value`` ("lead"), ``+-value`` alternating
    ("alternate"), ``-value`` but ``value`` in lane 0's 4-element vectors ("lane0"), or
    ``-value`` to ``value`` ("ramp")."""
    if kind == "lead":
        x = torch.zeros(1, 1, width, device=run_device())
        x[..., :4] = value
    elif kind == "alternate":
        x = torch.full((1, 1, width), value, device=run_device())
        x[..., ::2] = -value
    elif kind == "ramp":
        x = torch.linspace(-value, value, width, device=run_device()).view(1, 1, width)
    else:
        x = torch.full((1, 1, width), -value, device=run_device())
        x[..., (torch.arange(width, device=x.device) // 4) % 32 == 0] = value
    return x


def _extreme_case(kind, width, value, path, mark, weight=1.0, eps=1e-5, tag=""):
    return pytest.param(
        kind, width, value, weight, eps, path, marks=mark, id=f"{tag or kind}-{path}-{value:g}"
    )


# "lead": a lane's differences (5e18) or the lanes' about lane 0's mean (4e18) square past
# fp32's range unless scaled. "weight": a weight of 5e37 folds into rstd only within range.
# "tiny": squares near fp32's smallest normal, taken again unscaled. "bound": references at,
# below and above the bound past which a warp keeps its scale (2^55 at 1024, 2^54 at 4096).
# "single": one element a thread, whose warp merge still takes the lanes' factor.
_EXTREME_CASES = smoke_first(
    [
        _extreme_case("lead", 1024, 5e18, "warp", pytest.mark.smoke),
        _extreme_case("lead", 128, 4e18, "warp", pytest.mark.full),
        _extreme_case("alternate", 1024, 1.0, "warp", pytest.mark.full, weight=5e37, tag="weight"),
        _extreme_case(
            "alternate", 1024, 1.3e-19, "warp", pytest.mark.full, eps=2.0**-126, tag="tiny"
        ),
        _extreme_case(
            "alternate", 1024, 1.3e-20, "warp", pytest.mark.full, eps=2.0**-132, tag="tiny"
        ),
        _extreme_case("lane0", 1024, 2.0**55, "warp", pytest.mark.full, tag="bound"),
        _extreme_case("lane0", 1024, 2.0**56, "warp", pytest.mark.full, tag="bound"),
        _extreme_case("ramp", 3, 1.0, "held", pytest.mark.smoke, tag="single"),
        _extreme_case("alternate", 4003, 1e18, "staged", pytest.mark.full, tag="square-sum"),
        _extreme_case("alternate", 60000, 1e18, "split", pytest.mark.full, tag="square-sum"),
        _extreme_case("alternate", 4096, 1.0, "held", pytest.mark.full, weight=5e37, tag="weight"),
        *[
            _extreme_case(
                "alternate", width, 1.3e-21, path, pytest.mark.full, eps=2.0**-138, tag="tiny"
            )
            for width, path in [(1024, "warp"), (4096, "held")]
        ],
        _extreme_case(
            "alternate", 4096, 1.3e-19, "held", pytest.mark.full, eps=2.0**-126, tag="tiny"
        ),
        _extreme_case(
            "alternate", 4096, 1.3e-20, "held", pytest.mark.full, eps=2.0**-132, tag="tiny"
        ),
        _extreme_case("lane0", 1024, 2.0**54, "warp", pytest.mark.full, tag="bound"),
        *[
            _extreme_case("lane0", 4096, 2.0**e, "held", pytest.mark.full, tag="bound")
            for e in (53, 54, 55)
        ],
    ],
)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("kind, width, value, weight, eps, path", _EXTREME_CASES)
def test_group_norm_matches_float64_on_extreme_rows(kind, width, value, weight, eps, path) -> None:
    x = _extreme_row(kind, width, value)
    weight = torch.full((1,), weight, device=x.device)
    bias = torch.zeros(1, device=x.device)
    op = GroupNormFwdOp(num_groups=1, eps=eps)
    y = op(x, weight, bias)
    (built,) = op.iter_kernels()
    assert _path(built) == path
    expected = torch.nn.functional.group_norm(
        x.double(), 1, weight.double(), bias.double(), eps=eps
    )
    compare_outputs(y, expected.float(), normalization_verification("GroupNormFwdOp", x.dtype))


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
