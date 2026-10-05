import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.backend import BUILTIN
from tileops.kernels.norm import GroupNormKernel, GroupNormNoAffineKernel
from tileops.ops._signature_codegen import CheckError
from tileops.ops.norm.group_norm import GroupNormFwdOp
from workloads.device import run_device
from workloads.norm import GroupNormWorkload


class GroupNormTest(GroupNormWorkload, TestBase):
    pass


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
    assert len(op.built_kernels("group_norm")) == 1
    assert op.eval_roofline() == (
        7 * 2 * 16 * 16,
        (2 * 2 * 16 * 16 + 2 * 16) * torch.float16.itemsize,
    )

    run_case(2, 16, (4, 4), torch.float16)
    assert len(op.built_kernels("group_norm")) == 1

    run_case(3, 24, (2, 8), torch.bfloat16)
    assert len(op.built_kernels("group_norm")) == 2
    assert op.eval_roofline() == (
        7 * 3 * 24 * 16,
        (2 * 3 * 24 * 16 + 2 * 24) * torch.bfloat16.itemsize,
    )


@pytest.mark.cuda_only
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
    """One forward takes x plus the optional affine pair."""
    import inspect

    sig = inspect.signature(GroupNormFwdOp.forward)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["x", "weight", "bias"], f"got {params}"
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


class _MultiRowGroupNormKernel(GroupNormKernel):
    """Pin a four-row block, which the untuned default never picks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = {"block_m": 4, "threads": 128}


class _MultiRowGroupNormNoAffineKernel(GroupNormNoAffineKernel):
    """Pin a four-row block, which the untuned default never picks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = {"block_m": 4, "threads": 128}


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "n, c, spatial, g, affine",
    [
        # M = 9 rows of D = 256: the tail row block runs past M with aligned columns.
        (3, 24, (4, 8), 3, True),
        (3, 24, (4, 8), 3, False),
        # M = 9, D = 200: the tail row block and the column padding together.
        (3, 24, (5, 5), 3, True),
        (3, 24, (5, 5), 3, False),
        # M = 3 < block_m: the only block is a partial one.
        (1, 24, (4, 8), 3, False),
    ],
)
def test_group_norm_multi_row_block_runs_past_m(
    n: int, c: int, spatial: tuple, g: int, affine: bool
) -> None:
    """A block of several rows stays correct where it runs past the last row.

    ``block_m`` above one is an autotune candidate; the op reaches it through dispatch
    with the key's implementation pinned to that block.
    """
    key, cls = (
        ("group_norm", _MultiRowGroupNormKernel)
        if affine
        else ("group_norm_no_affine", _MultiRowGroupNormNoAffineKernel)
    )
    test = GroupNormTest(n, c, spatial, g, torch.float16)
    x, weight, bias = test.gen_inputs()
    op = GroupNormFwdOp(num_groups=g, kernel_map={key: cls}, target=BUILTIN)
    test.check(op, *((x, weight, bias) if affine else (x,)))
    (kernel,) = op.built_kernels("group_norm").values()
    assert type(kernel) is cls and kernel.config["block_m"] == 4


@pytest.mark.smoke
@pytest.mark.parametrize(
    "passes_affine, key", [(True, "group_norm"), (False, "group_norm_no_affine")]
)
def test_each_region_selects_its_one_implementation(passes_affine: bool, key: str) -> None:
    """A call passing the affine pair takes the affine program, any other the plain one."""
    from tileops.kernels.norm.call_spec import GroupNormCall

    call = GroupNormCall(
        arch=90, sm_count=132, c=64, spatial=64, num_groups=8, passes_affine=passes_affine
    )
    assert GroupNormFwdOp(num_groups=8).select_implementation("group_norm", call) == key
