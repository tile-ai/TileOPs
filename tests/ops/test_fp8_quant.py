import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.quantization import FP8QuantPerTokenGroupFwdOp
from workloads.device import run_device
from workloads.quantization.fp8_quant import FP8QuantWorkload


class FP8QuantTest(FP8QuantWorkload, TestBase):
    pass


class FP8QuantFixture(FixtureBase):
    PARAMS = [
        (
            "shape, group_size, dtype, tune",
            [
                pytest.param((8192, 64), 64, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param((8192, 64), 64, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param((8192, 64), 64, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param((2, 64, 512), 128, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param((4096, 128), 128, torch.float32, False, marks=pytest.mark.full),
                pytest.param((16384, 32), 32, torch.float32, False, marks=pytest.mark.full),
                pytest.param((1024, 4, 64), 64, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@FP8QuantFixture
def test_fp8_quant_op(shape, group_size: int, dtype: torch.dtype, tune: bool) -> None:
    test = FP8QuantTest(shape, group_size, dtype)
    op = FP8QuantPerTokenGroupFwdOp(group_size)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_fp8_quant_writes_out_q_and_returns_it() -> None:
    test = FP8QuantTest((64, 256), 128, torch.bfloat16, out_q=True)
    x, out_q = test.gen_inputs()
    q, _scale, _output = FP8QuantPerTokenGroupFwdOp(128)(x, out_q)
    assert q is out_q
    test.check(FP8QuantPerTokenGroupFwdOp(128), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "layout, reason",
    [
        ({"column_major_scales": True}, "does not serve"),
        ({"use_ue8m0": True}, "does not serve"),
        ({"use_ue8m0": True, "scale_dtype": torch.float8_e8m0fnu}, "float32 scales"),
        ({"inplace": True}, "in place"),
    ],
    ids=["column-major", "ue8m0", "e8m0-scales", "inplace"],
)
def test_fp8_quant_refuses_what_no_kernel_serves(layout, reason) -> None:
    from tileops.backend import OpNotAvailableError

    x = torch.randn(4, 128, dtype=torch.bfloat16, device=run_device())
    with pytest.raises(OpNotAvailableError, match=reason):
        FP8QuantPerTokenGroupFwdOp(128, **layout)(x)


@pytest.mark.smoke
def test_fp8_quant_reads_tma_alignment_only_for_column_major_scales() -> None:
    test = FP8QuantTest((64, 256), 128, torch.bfloat16)
    test.check(FP8QuantPerTokenGroupFwdOp(128, tma_aligned_scales=True), *test.gen_inputs())


@pytest.mark.smoke
def test_fp8_quant_serves_new_token_counts_from_one_kernel() -> None:
    op = FP8QuantPerTokenGroupFwdOp(128)
    for tokens in (3, 70):
        test = FP8QuantTest((tokens, 256), 128, torch.bfloat16)
        test.check(op, *test.gen_inputs())
    assert len(list(op.iter_kernels())) == 1


@pytest.mark.smoke
def test_fp8_quant_refuses_a_group_past_the_register_budget() -> None:
    from tileops.backend import OpNotAvailableError

    x = torch.randn(4, 384, dtype=torch.bfloat16, device=run_device())
    with pytest.raises(OpNotAvailableError, match="register budget"):
        FP8QuantPerTokenGroupFwdOp(384)(x)
