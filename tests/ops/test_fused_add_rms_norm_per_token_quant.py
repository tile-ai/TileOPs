"""Tests for the fused residual add, RMSNorm and dynamic per-token quantization."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.norm.fused_add_rms_norm_per_token_quant import FusedAddRMSNormPerTokenQuantFwdOp
from workloads.norm import FusedAddRMSNormPerTokenQuantWorkload
from workloads.numerics import compare_outputs

FP8, INT8 = torch.float8_e4m3fn, torch.int8


class FusedAddRMSNormPerTokenQuantTest(FusedAddRMSNormPerTokenQuantWorkload, TestBase):
    pass


class FusedAddRMSNormPerTokenQuantFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype, quant_dtype, scale_ub",
            [
                # A row of up to 4096 columns runs 128 threads, a wider one 512.
                pytest.param((64, 4096), torch.bfloat16, FP8, None, marks=pytest.mark.smoke),
                pytest.param((64, 8192), torch.float16, INT8, None, marks=pytest.mark.smoke),
                # A row the threads' accesses do not cover evenly skips the accesses past
                # its end: single elements at 257, 16-byte accesses at 3000, 8-byte at 4100,
                # 4-byte at 4098.
                pytest.param((1, 257), torch.bfloat16, INT8, None, marks=pytest.mark.smoke),
                pytest.param((33, 3000), torch.bfloat16, INT8, None, marks=pytest.mark.full),
                pytest.param((3, 4100), torch.float16, FP8, None, marks=pytest.mark.full),
                pytest.param((2, 4098), torch.bfloat16, FP8, None, marks=pytest.mark.full),
                # Leading axes flatten into rows and come back on every output.
                pytest.param((2, 7, 4096), torch.float16, FP8, None, marks=pytest.mark.full),
                # A bound below every row's amax: each scale is the bound, and the codes
                # past it saturate.
                pytest.param((16, 4096), torch.bfloat16, FP8, 0.5, marks=pytest.mark.full),
            ],
        ),
    ]


@FusedAddRMSNormPerTokenQuantFixture
def test_fused_add_rms_norm_per_token_quant(shape, dtype, quant_dtype, scale_ub) -> None:
    test = FusedAddRMSNormPerTokenQuantTest(shape, dtype, quant_dtype, scale_ub=scale_ub)
    op = FusedAddRMSNormPerTokenQuantFwdOp(quant_dtype=quant_dtype, scale_ub=scale_ub)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("quant_dtype", [FP8, INT8])
def test_a_row_summing_to_zero_takes_the_floor_scale(quant_dtype) -> None:
    """``residual = -x`` zeroes a row: its scale is the format's floor and its codes zero."""
    workload = FusedAddRMSNormPerTokenQuantWorkload((4, 4096), torch.bfloat16, quant_dtype)
    x, residual, weight = workload.gen_inputs()
    residual[0] = -x[0]
    got = FusedAddRMSNormPerTokenQuantFwdOp(quant_dtype=quant_dtype)(x, residual, weight)
    compare_outputs(got, workload.ref_program(x, residual, weight), workload.verification(x))
    floor = 1.0 / (448.0 * 512.0) if quant_dtype == FP8 else torch.finfo(torch.float32).eps
    assert got[1][0].item() == pytest.approx(floor, rel=0)
    assert not got[0][0].float().any()


@pytest.mark.smoke
@pytest.mark.parametrize("quant_dtype", [FP8, INT8])
def test_nonfinite_values_take_vllms_codes(quant_dtype) -> None:
    """A NaN is skipped by the amax and saturates as vLLM's conversion does; an infinite sum
    turns its row's normalized values into NaN and zeros."""
    workload = FusedAddRMSNormPerTokenQuantWorkload((4, 1024), torch.float16, quant_dtype)
    x, residual, weight = workload.gen_inputs()
    x[0, 5] = float("nan")
    x[1, 9] = float("inf")
    got = FusedAddRMSNormPerTokenQuantFwdOp(quant_dtype=quant_dtype)(x, residual, weight)
    compare_outputs(got, workload.ref_program(x, residual, weight), workload.verification(x))
