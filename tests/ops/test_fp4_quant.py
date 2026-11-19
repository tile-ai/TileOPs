import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.quantization import FP4QuantPerGroupFwdOp
from workloads.device import run_device
from workloads.quantization.quantize import (
    FP4QuantTiesWorkload,
    FP4QuantViewWorkload,
    FP4QuantWorkload,
)


class FP4QuantTest(FP4QuantWorkload, TestBase):
    pass


class FP4QuantTiesTest(FP4QuantTiesWorkload, TestBase):
    pass


class FP4QuantViewTest(FP4QuantViewWorkload, TestBase):
    pass


@pytest.mark.parametrize(
    "shape, block_size, inplace",
    [
        # DeepSeek-V4-Flash's indexer queries, quantized and fake-quantized.
        pytest.param((64, 64, 128), 32, False, marks=pytest.mark.smoke),
        pytest.param((64, 64, 128), 32, True, marks=pytest.mark.smoke),
        # Its compressed keys; a last block of rows the CTA does not fill.
        pytest.param((3000, 128), 32, True, marks=pytest.mark.full),
        pytest.param((3000, 128), 32, False, marks=pytest.mark.full),
        # Other power-of-two block sizes.
        pytest.param((256, 256), 16, False, marks=pytest.mark.full),
        pytest.param((256, 256), 128, True, marks=pytest.mark.full),
        # Runs of 4 and 2 codes, packed into a uint16 and a uint8; 12's odd factor puts each
        # block in one thread.
        pytest.param((64, 48), 12, False, marks=pytest.mark.full),
        pytest.param((64, 64), 2, False, marks=pytest.mark.full),
        # An odd block size, fake-quantized one column a run.
        pytest.param((64, 63), 3, True, marks=pytest.mark.full),
        # An input large enough for the wide launch, in both modes.
        pytest.param((512, 64, 128), 32, False, marks=pytest.mark.full),
        pytest.param((512, 64, 128), 32, True, marks=pytest.mark.full),
    ],
)
def test_fp4_quant_op(shape, block_size, inplace) -> None:
    test = FP4QuantTest(shape, block_size, inplace)
    test.check(FP4QuantPerGroupFwdOp(block_size, inplace), *test.gen_inputs())


@pytest.mark.smoke
def test_fp4_quant_runs_after_a_tuning_request() -> None:
    test = FP4QuantTest((64, 128), 32, False)
    op = FP4QuantPerGroupFwdOp()
    op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "inplace",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.smoke)],
)
def test_fp4_quant_rounds_ties_to_even_and_keeps_signed_zeros(inplace) -> None:
    """Values on the midpoint of two e2m1 values round to the even code, a negative value
    rounding to zero keeps its sign, and an all-zero block takes the floor."""
    test = FP4QuantTiesTest((256, 128), 32, inplace)
    test.check(FP4QuantPerGroupFwdOp(32, inplace), *test.gen_inputs())


@pytest.mark.parametrize(
    "wide_shape, index",
    [
        # Columns of wider rows: rows 160 apart, fake-quantized without a copy.
        pytest.param((50, 160), (..., slice(0, 128)), marks=pytest.mark.smoke, id="leading"),
        # Rows not evenly spaced go through a contiguous copy.
        pytest.param(
            (3, 30, 128), (slice(None), slice(0, 20)), marks=pytest.mark.full, id="uneven-rows"
        ),
        # Rows of more blocks than the grid's y extent go through a contiguous copy.
        pytest.param(
            (2, 65537 * 32), (..., slice(0, 65536 * 32)), marks=pytest.mark.full, id="past-grid-y"
        ),
    ],
)
def test_fp4_quant_fake_quantizes_a_strided_view_in_place(wide_shape, index) -> None:
    """Only the view's elements change, and the call returns the view itself."""
    x_shape = torch.empty(wide_shape, device="meta")[index].shape
    test = FP4QuantViewTest(wide_shape, index, x_shape, 32, True)
    op = FP4QuantPerGroupFwdOp(32, inplace=True)
    test.check(op, *test.gen_inputs())
    (x,) = test.gen_inputs()
    wide = x._base
    before = wide.clone()
    assert op(x)[2] is x
    kept = torch.ones_like(wide, dtype=torch.bool)
    kept[index] = False
    assert torch.equal(wide[kept], before[kept])


@pytest.mark.smoke
def test_fp4_quant_rejects_an_input_other_than_bfloat16() -> None:
    x = torch.randn(4, 64, dtype=torch.float16, device=run_device())
    with pytest.raises(ValueError, match="bfloat16"):
        FP4QuantPerGroupFwdOp()(x)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_fp4_quant_serves_new_token_counts_from_one_kernel() -> None:
    op = FP4QuantPerGroupFwdOp(32, inplace=True)
    for tokens in (3, 70):
        test = FP4QuantTest((tokens, 128), 32, True)
        test.check(op, *test.gen_inputs())
    assert len(list(op.iter_kernels())) == 1


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "block_size, inplace, reason",
    [
        (3, False, "two blocks"),
        (192, True, "register budget"),
    ],
    ids=["odd-block", "past-register-budget"],
)
def test_fp4_quant_refuses_what_no_kernel_serves(block_size, inplace, reason) -> None:
    from tileops.backend import OpNotAvailableError

    x = torch.randn(4, 2 * block_size, dtype=torch.bfloat16, device=run_device())
    before = x.clone()
    with pytest.raises(OpNotAvailableError, match=reason):
        FP4QuantPerGroupFwdOp(block_size, inplace)(x)
    assert torch.equal(x, before)
