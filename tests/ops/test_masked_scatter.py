import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.indexing import MaskedScatterFwdOp
from workloads.device import run_device
from workloads.indexing import MaskedScatterWorkload
from workloads.workload_base import manifest_call


class MaskedScatterTest(MaskedScatterWorkload, TestBase):
    pass


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
# One case per dtype and per path: whole rows a block with the block offset read from the
# mask, column tiles of a row wider than a block with a tail, a row width off the 16-byte
# vector, broadcasts, and an elementwise mask past 4 * block_runs * run_len * itemsize runs
# (4 * 2048 * 1 * 2 = 16384 for bf16), which reads the block offset from a count pass.
_CASES = (
    pytest.param([96, 2560], [96, 1], [96, 2560], "bfloat16", marks=_SMOKE, id="bf16-rows"),
    pytest.param([40, 4096], [40, 1], [50, 4096], "float16", marks=_SMOKE, id="fp16-rows"),
    pytest.param([33, 4096], [33, 1], [33 * 4096], "float32", marks=_SMOKE, id="fp32-rows"),
    pytest.param([16385], [16385], [16385], "bfloat16", marks=_FULL, id="counted-elements"),
    pytest.param([3, 20000], [3, 1], [3, 20000], "float32", marks=_FULL, id="col-tiles"),
    pytest.param([9, 7], [9, 1], [70], "float32", marks=_FULL, id="odd-width"),
    pytest.param([1, 64], [12, 1], [12 * 64], "bfloat16", marks=_FULL, id="input-broadcast"),
    pytest.param(
        [6, 5, 64], [5, 1], [6 * 5 * 64], "float16", marks=_FULL, id="mask-lead-broadcast"
    ),
    # A run of more than 65535 column tiles, the grid's y limit, takes several passes.
    pytest.param(
        [1, 65535 * 16384 + 24],
        [1, 1],
        [65535 * 16384 + 24],
        "float16",
        marks=_FULL,
        id="widest-run",
    ),
)


@pytest.mark.parametrize("A, M, R, dtype", _CASES)
def test_masked_scatter(A, M, R, dtype) -> None:
    call = manifest_call("MaskedScatterFwdOp", {"T": dtype}, A=A, M=M, R=R)
    test = MaskedScatterTest(call)
    test.check(MaskedScatterFwdOp(**call.arguments()), *test.gen_inputs())


@pytest.mark.smoke
def test_masked_scatter_counts_nonzero_mask_bytes() -> None:
    """A bool stored as a byte other than 1 counts once, as torch reads a row mask."""
    # Two float32 rows a block: the blocks past the first read 16-byte chunks of mask bytes.
    pattern = torch.tensor([2, 1, 0, 3], dtype=torch.uint8, device=run_device())
    bytes_ = pattern.repeat(10)
    call = manifest_call(
        "MaskedScatterFwdOp", {"T": "float32"}, A=[40, 4096], M=[40, 1], R=[30, 4096]
    )
    test = MaskedScatterTest(call)
    input, _, source = test.gen_inputs()
    test.check(MaskedScatterFwdOp(), input, bytes_.view(torch.bool).view(40, 1), source)
