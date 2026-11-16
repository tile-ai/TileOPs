import math

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.indexing import MaskedScatterFwdOp
from workloads.indexing import MaskedScatterWorkload
from workloads.workload_base import manifest_call


class MaskedScatterTest(MaskedScatterWorkload, TestBase):
    pass


def _test(
    input_shape: tuple,
    mask_shape: tuple,
    dtype: str,
    source_shape: "tuple | None" = None,
    **inputs,
) -> MaskedScatterTest:
    """A call whose source defaults to a flat element for every position of the output."""
    if source_shape is None:
        source_shape = (math.prod(torch.broadcast_shapes(input_shape, mask_shape)),)
    call = manifest_call(
        "MaskedScatterFwdOp",
        {"T": dtype},
        A=list(input_shape),
        M=list(mask_shape),
        R=list(source_shape),
    )
    return MaskedScatterTest(call, **inputs)


# The smoke shape of each dtype; the tests of one input form reuse the bfloat16 one.
_BF16 = ((37, 2560), (37, 1), "bfloat16")
_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
_CASES = (
    # A row mask, as a multimodal model writes image rows: row and column tile tails.
    pytest.param(*_BF16, marks=_SMOKE, id="row-mask-bf16"),
    pytest.param((37, 1000), (37, 1), "float16", marks=_SMOKE, id="row-mask-fp16"),
    pytest.param((37, 1000), (37, 1), "float32", marks=_SMOKE, id="row-mask-fp32"),
    # Rows of 7 elements: no 16-byte vector divides them.
    pytest.param((5, 7), (5, 1), "float16", marks=_FULL, id="odd-row"),
    # A mask of input's shape: rows of one element, four a thread in the tile scan.
    pytest.param((37, 100), (37, 100), "float32", marks=_FULL, id="full-mask"),
    # The mask broadcasts along a leading axis of input, and input along the mask's rows.
    pytest.param((3, 5, 64), (5, 1), "bfloat16", marks=_FULL, id="mask-leading-broadcast"),
    pytest.param((1, 64), (37, 1), "bfloat16", marks=_FULL, id="input-broadcast"),
    # The most mask rows the implementation serves, 2**31 - 2**16.
    pytest.param((2**16, 1, 1), (1, 2**15 - 1, 1), "bfloat16", marks=_FULL, id="most-rows"),
    # More than 2**31 elements: input and output offsets past int32, and several passes of the
    # tile-offset scan.
    pytest.param((2**17 + 3, 2**14), (2**17 + 3, 1), "bfloat16", marks=_FULL, id="int64-offsets"),
    # One row of 65536 column tiles: more than a CUDA grid's y or z axis holds.
    pytest.param((33554432,), (), "bfloat16", marks=_FULL, id="one-wide-row"),
)


class MaskedScatterFixture(FixtureBase):
    PARAMS = [("input_shape, mask_shape, dtype", _CASES)]


@MaskedScatterFixture
def test_masked_scatter(input_shape: tuple, mask_shape: tuple, dtype: str) -> None:
    test = _test(input_shape, mask_shape, dtype)
    test.check(MaskedScatterFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
def test_masked_scatter_without_true_rows_reads_no_source() -> None:
    """A mask with no true row copies input and reads no source, so an empty source serves
    it."""
    test = _test(*_BF16, no_true=True, source_extra=0)
    test.check(MaskedScatterFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_masked_scatter_serves_new_row_widths_from_one_kernel() -> None:
    """A mask of extent 1 on every axis makes the whole output one row, so the row width
    changes with the token count; the width is dynamic, and one kernel serves both."""
    op = MaskedScatterFwdOp()
    for tokens in (37, 41):
        test = _test((tokens, 2560), (), "bfloat16")
        test.check(op, *test.gen_inputs())
    assert len(list(op.iter_kernels())) == 1


@pytest.mark.smoke
@pytest.mark.parametrize("source_extra", [0, 100], ids=["packed", "partial-last-row"])
def test_masked_scatter_source_length(source_extra: int) -> None:
    """A source of exactly the true rows' elements, and one with part of a row more."""
    test = _test(*_BF16, source_extra=source_extra)
    test.check(MaskedScatterFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
def test_masked_scatter_reads_any_nonzero_byte_as_true() -> None:
    """A bool tensor may hold true as any nonzero byte; each one takes one source element."""
    test = _test(*_BF16, true_byte=255)
    test.check(MaskedScatterFwdOp(), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "input_shape, mask_shape, reason",
    [
        pytest.param(
            (2**16, 1, 1), (1, 2**15, 1), "2\\*\\*31 - 2\\*\\*16 mask rows", marks=_SMOKE, id="rows"
        ),
        pytest.param(
            (1, 2**20 + 1), (2**20, 1), "2\\*\\*31 - 1 copy tiles", marks=_FULL, id="tiles"
        ),
        pytest.param((2**31,), (), "2\\*\\*31 - 2\\*\\*16 mask rows", marks=_FULL, id="row-width"),
    ],
)
def test_masked_scatter_refuses_past_its_limits(
    input_shape: tuple, mask_shape: tuple, reason: str
) -> None:
    """The implementation keeps rows and columns, padded to a whole tile, in int32 and launches
    one CUDA grid axis of copy tiles. A call of 2**31 mask rows, of one row of 2**31 elements,
    or of 2**31 copy tiles is refused before an output is allocated."""
    from tileops.backend import OpNotAvailableError

    test = _test(input_shape, mask_shape, "bfloat16", source_shape=(1,))
    with pytest.raises(OpNotAvailableError, match=reason):
        MaskedScatterFwdOp()(*test.gen_inputs())
