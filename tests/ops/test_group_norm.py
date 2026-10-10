import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
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
                # Wide groups: Stable Diffusion's 64 x 64 latent (40960 a group, held in
                # registers), an SDXL latent no block holds, a bf16 group of 262144 cut
                # across blocks, and an fp32 row past SM89's shared memory.
                pytest.param(2, 320, (64, 64), 32, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(2, 320, (152, 104), 32, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1, 128, (256, 256), 32, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 128, (50, 125), 32, torch.float32, False, marks=pytest.mark.full),
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
