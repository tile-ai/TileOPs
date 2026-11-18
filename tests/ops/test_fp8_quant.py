import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.quantization import FP8QuantPerTokenGroupFwdOp
from workloads.device import run_device
from workloads.quantization.fp8_quant import (
    FP8QuantScaleFloorWorkload,
    FP8QuantTiesWorkload,
    FP8QuantViewWorkload,
    FP8QuantWorkload,
)


class FP8QuantTest(FP8QuantWorkload, TestBase):
    pass


class FP8QuantTiesTest(FP8QuantTiesWorkload, TestBase):
    pass


class FP8QuantViewTest(FP8QuantViewWorkload, TestBase):
    pass


class FP8QuantScaleFloorTest(FP8QuantScaleFloorWorkload, TestBase):
    pass


class FP8QuantFixture(FixtureBase):
    PARAMS = [
        (
            "shape, group_size, dtype",
            [
                pytest.param((8192, 64), 64, torch.float16, marks=pytest.mark.smoke),
                pytest.param((8192, 64), 64, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param((8192, 64), 64, torch.float32, marks=pytest.mark.smoke),
                pytest.param((2, 64, 512), 128, torch.bfloat16, marks=pytest.mark.full),
                pytest.param((4096, 128), 128, torch.float32, marks=pytest.mark.full),
                pytest.param((16384, 32), 32, torch.float32, marks=pytest.mark.full),
                pytest.param((1024, 4, 64), 64, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@FP8QuantFixture
def test_fp8_quant_op(shape, group_size: int, dtype: torch.dtype) -> None:
    test = FP8QuantTest(shape, group_size, dtype)
    test.check(FP8QuantPerTokenGroupFwdOp(group_size), *test.gen_inputs())


@pytest.mark.smoke
def test_fp8_quant_writes_out_q_and_returns_it() -> None:
    test = FP8QuantTest((64, 256), 128, torch.bfloat16, out_q=True)
    x, out_q = test.gen_inputs()
    q, _scale, _output = FP8QuantPerTokenGroupFwdOp(128)(x, out_q)
    assert q is out_q
    test.check(FP8QuantPerTokenGroupFwdOp(128), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_fp8_quant_refuses_column_major_scales() -> None:
    from tileops.backend import OpNotAvailableError

    x = torch.randn(4, 128, dtype=torch.bfloat16, device=run_device())
    with pytest.raises(OpNotAvailableError, match="column-major"):
        FP8QuantPerTokenGroupFwdOp(128, column_major_scales=True)(x)


# DeepSeek-V4-Flash's act_quant: the 1e-4 floor and power-of-two scales.
_UE8M0 = {"eps": 1e-4, "use_ue8m0": True}


@pytest.mark.parametrize(
    "shape, group_size, dtype, scale_dtype",
    [
        # A DeepSeek-V4-Flash GEMM input with e8m0 scales, and the float32 ones.
        pytest.param(
            (256, 4096), 128, torch.bfloat16, torch.float8_e8m0fnu, marks=pytest.mark.smoke
        ),
        pytest.param((64, 1024), 128, torch.float16, torch.float8_e8m0fnu, marks=pytest.mark.smoke),
        pytest.param((64, 1024), 128, torch.float32, torch.float32, marks=pytest.mark.smoke),
        pytest.param((256, 4096), 128, torch.bfloat16, torch.float32, marks=pytest.mark.full),
        # A last block of rows the block does not fill.
        pytest.param(
            (3000, 2048), 128, torch.bfloat16, torch.float8_e8m0fnu, marks=pytest.mark.full
        ),
        # One token, and leading axes folded into the rows.
        pytest.param((1, 4096), 128, torch.bfloat16, torch.float8_e8m0fnu, marks=pytest.mark.full),
        pytest.param((3, 5, 512), 64, torch.bfloat16, torch.float32, marks=pytest.mark.full),
    ],
)
def test_fp8_quant_rounds_scales_up_to_a_power_of_two(
    shape, group_size, dtype, scale_dtype
) -> None:
    test = FP8QuantTest(shape, group_size, dtype, scale_dtype=scale_dtype, **_UE8M0)
    op = FP8QuantPerTokenGroupFwdOp(group_size, scale_dtype=scale_dtype, **_UE8M0)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "dtype, inplace",
    [
        pytest.param(torch.bfloat16, False, marks=pytest.mark.smoke),
        pytest.param(torch.bfloat16, True, marks=pytest.mark.smoke),
        pytest.param(torch.float16, False, marks=pytest.mark.smoke),
        pytest.param(torch.float32, True, marks=pytest.mark.smoke),
    ],
)
def test_fp8_quant_ue8m0_rounds_ties_to_even_and_keeps_signed_zeros(dtype, inplace) -> None:
    """Codes on the midpoint of two e4m3 values round to the even one, a negative value
    rounding to zero keeps its sign, and an all-zero group takes the ``eps`` floor."""
    test = FP8QuantTiesTest((64, 512), 128, dtype, inplace=inplace, **_UE8M0)
    test.check(FP8QuantPerTokenGroupFwdOp(128, inplace=inplace, **_UE8M0), *test.gen_inputs())


@pytest.mark.parametrize(
    "dtype, eps, scale_dtype, inplace",
    [
        pytest.param(torch.bfloat16, 0.0, torch.float32, False, marks=pytest.mark.smoke),
        pytest.param(torch.float32, 1e-36, torch.float8_e8m0fnu, False, marks=pytest.mark.smoke),
        pytest.param(torch.float16, 0.0, torch.float8_e8m0fnu, False, marks=pytest.mark.smoke),
        pytest.param(torch.bfloat16, 1e-36, torch.float32, True, marks=pytest.mark.full),
        pytest.param(torch.float32, 0.0, torch.float32, True, marks=pytest.mark.full),
    ],
)
def test_fp8_quant_ue8m0_floors_scales_at_two_to_the_minus_126(
    dtype, eps, scale_dtype, inplace
) -> None:
    """A group whose ``max(amax, eps) / 448`` is subnormal or zero takes the scale
    ``2 ** -126``, as ``act_quant`` gives a subnormal one."""
    params = {"eps": eps, "use_ue8m0": True, "scale_dtype": scale_dtype, "inplace": inplace}
    test = FP8QuantScaleFloorTest((80, 512), 128, dtype, **params)
    op = FP8QuantPerTokenGroupFwdOp(128, **params)
    test.check(op, *test.gen_inputs())
    if not inplace:
        _, scale, _ = op(*test.gen_inputs())
        assert scale.float().min().item() == 2.0**-126


@pytest.mark.parametrize(
    "shape, dtype",
    [
        # DeepSeek-V4-Flash's KV rows, 448 columns per 64.
        pytest.param((4096, 448), torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param((37, 448), torch.float16, marks=pytest.mark.smoke),
        pytest.param((37, 448), torch.float32, marks=pytest.mark.smoke),
        pytest.param((1, 448), torch.bfloat16, marks=pytest.mark.full),
    ],
)
def test_fp8_quant_fake_quantizes_in_place(shape, dtype) -> None:
    test = FP8QuantTest(shape, 64, dtype, inplace=True, **_UE8M0)
    op = FP8QuantPerTokenGroupFwdOp(64, inplace=True, **_UE8M0)
    test.check(op, *test.gen_inputs())
    (x,) = test.gen_inputs()
    q, scale, output = op(x)
    assert q is None and scale is None and output is x


@pytest.mark.parametrize(
    "wide_shape, index",
    [
        # DeepSeek-V4-Flash's kv[..., :448]: rows 512 apart, quantized without a copy.
        pytest.param((3, 50, 512), (..., slice(0, 448)), marks=pytest.mark.smoke, id="leading"),
        pytest.param((70, 512), (..., slice(64, 512)), marks=pytest.mark.full, id="trailing"),
        # Rows not evenly spaced go through a contiguous copy.
        pytest.param(
            (3, 60, 448), (slice(None), slice(0, 50)), marks=pytest.mark.full, id="uneven-rows"
        ),
    ],
)
def test_fp8_quant_fake_quantizes_a_strided_view_in_place(wide_shape, index) -> None:
    """Only the view's elements change, and the call returns the view itself."""
    x_shape = torch.empty(wide_shape, device="meta")[index].shape
    test = FP8QuantViewTest(wide_shape, index, x_shape, 64, torch.bfloat16, inplace=True, **_UE8M0)
    op = FP8QuantPerTokenGroupFwdOp(64, inplace=True, **_UE8M0)
    test.check(op, *test.gen_inputs())
    (x,) = test.gen_inputs()
    wide = x._base
    before = wide.clone()
    assert op(x)[2] is x
    kept = torch.ones_like(wide, dtype=torch.bool)
    kept[index] = False
    assert torch.equal(wide[kept], before[kept])


@pytest.mark.smoke
def test_fp8_quant_reads_tma_alignment_only_for_column_major_scales() -> None:
    test = FP8QuantTest((64, 256), 128, torch.bfloat16)
    test.check(FP8QuantPerTokenGroupFwdOp(128, tma_aligned_scales=True), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_fp8_quant_serves_new_token_counts_from_one_kernel() -> None:
    op = FP8QuantPerTokenGroupFwdOp(128)
    for tokens in (3, 70):
        test = FP8QuantTest((tokens, 256), 128, torch.bfloat16)
        test.check(op, *test.gen_inputs())
    assert len(list(op.iter_kernels())) == 1


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_fp8_quant_refuses_a_group_past_the_register_budget() -> None:
    from tileops.backend import OpNotAvailableError

    x = torch.randn(4, 384, dtype=torch.bfloat16, device=run_device())
    with pytest.raises(OpNotAvailableError, match="register budget"):
        FP8QuantPerTokenGroupFwdOp(384)(x)
