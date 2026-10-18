import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.kernels.transform import HadamardTransformCall
from tileops.transform import HadamardTransformFwdOp
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.transform import HadamardTransformWorkload


class HadamardTransformTest(HadamardTransformWorkload, TestBase):
    pass


# One case per kernel and per end of its range.
_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
_CASES = (
    pytest.param((37,), 32, torch.bfloat16, 1, marks=_SMOKE, id="warp-rows-lower-tail"),
    pytest.param((3, 5), 128, torch.float16, 1, marks=_SMOKE, id="warp-rows-fp16"),
    pytest.param((3,), 1024, torch.bfloat16, 1, marks=_SMOKE, id="exchange-lower"),
    pytest.param((300,), 4096, torch.float16, 1, marks=_SMOKE, id="exchange-four-warps"),
    pytest.param((4, 3), 2560, torch.float16, 20, marks=_SMOKE, id="base-exchange"),
    pytest.param((5,), 1, torch.bfloat16, 1, marks=_SMOKE, id="shared-n1"),
    pytest.param((3,), 12, torch.bfloat16, 12, marks=_SMOKE, id="shared-base12-m1"),
    pytest.param((4,), 20 * 256, torch.bfloat16, 20, marks=_SMOKE, id="shared-base-mma"),
    pytest.param((0,), 4096, torch.bfloat16, 1, marks=_SMOKE, id="empty-batch"),
    pytest.param((), 256, torch.bfloat16, 1, marks=_SMOKE, id="rank-1"),
    pytest.param((9,), 512, torch.bfloat16, 1, marks=_FULL, id="warp-rows-upper"),
    pytest.param((140,), 2048, torch.float16, 1, marks=_FULL, id="exchange-odd"),
    pytest.param((200,), 8192, torch.bfloat16, 1, marks=_FULL, id="exchange-eight-slots"),
    pytest.param((3,), 16384, torch.float16, 1, marks=_FULL, id="exchange-scalar-store"),
    pytest.param((2,), 32768, torch.bfloat16, 1, marks=_FULL, id="exchange-upper"),
    pytest.param((7,), 12 * 32, torch.bfloat16, 12, marks=_FULL, id="base-exchange-12"),
    pytest.param((3,), 28 * 128, torch.float16, 28, marks=_FULL, id="base-exchange-28"),
    pytest.param((5,), 40 * 64, torch.bfloat16, 40, marks=_FULL, id="base-exchange-widest"),
    pytest.param((7,), 8, torch.float16, 1, marks=_FULL, id="shared-no-mma"),
    pytest.param((3,), 56, torch.float16, 28, marks=_FULL, id="shared-base28-m2"),
    pytest.param((2,), 40960, torch.bfloat16, 40, marks=_FULL, id="shared-base40-widest"),
    pytest.param((2,), 65536, torch.bfloat16, 1, marks=_FULL, id="multi-pass-bits"),
    pytest.param((2,), 12 * 8192, torch.float16, 12, marks=_FULL, id="multi-pass-base"),
    pytest.param((1,), 20 * 65536, torch.bfloat16, 20, marks=_FULL, id="multi-pass-both"),
    pytest.param((1,), 28 * 4096, torch.float16, 28, marks=_FULL, id="multi-pass-base-28"),
    pytest.param((1,), 40 * 16384, torch.bfloat16, 40, marks=_FULL, id="multi-pass-base-40"),
)


class HadamardTransformFixture(FixtureBase):
    PARAMS = [("batch_shape, n, dtype, base_order", _CASES)]


@HadamardTransformFixture
def test_hadamard_transform(
    batch_shape: tuple, n: int, dtype: torch.dtype, base_order: int
) -> None:
    test = HadamardTransformTest(batch_shape, n, dtype, base_order)
    test.check(HadamardTransformFwdOp(base_order), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("n, base_order", [(128, 1), (12, 12)], ids=["vector-rows", "scalar-rows"])
def test_hadamard_transform_unaligned_storage(n: int, base_order: int) -> None:
    """An input whose storage starts off a 16-byte boundary reads the same rows."""
    rows = 6
    flat = torch.randn(rows * n + 1, device=run_device(), dtype=torch.bfloat16)
    x = flat[1:].view(rows, n)

    got = HadamardTransformFwdOp(base_order)(x)

    workload = HadamardTransformWorkload((rows,), n, x.dtype, base_order)
    compare_outputs(got, workload.ref_program(x), workload.verification(x))


@pytest.mark.in_tree_kernels
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("base_order", [1, 12, 20, 28, 40])
def test_every_row_length_has_a_kernel(base_order: int) -> None:
    """Every row the manifest serves, base_order times 2**0 through 2**28, has an implementation."""
    op = HadamardTransformFwdOp(base_order)
    for exponent in range(29):
        call = HadamardTransformCall(n=base_order << exponent, base_order=base_order)
        op.key_for("hadamard_transform", call)
