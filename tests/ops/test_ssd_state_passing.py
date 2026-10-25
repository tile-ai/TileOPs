import pytest
import torch

from tests.workload_test_base import TestBase
from tileops._ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import SSDStatePassingUnitWorkload


class PassingTest(SSDStatePassingUnitWorkload, TestBase):
    pass


@pytest.mark.parametrize(
    "chunks,width,dtype,initial,period,tune",
    [
        pytest.param(
            32768, 32, torch.float32, True, 0, False, id="long-fp32", marks=pytest.mark.smoke
        ),
        pytest.param(
            32768, 32, torch.float16, True, 0, False, id="long-fp16", marks=pytest.mark.smoke
        ),
        pytest.param(
            32768, 32, torch.bfloat16, True, 0, False, id="long-bf16", marks=pytest.mark.smoke
        ),
        pytest.param(
            257, 128, torch.float32, False, 251, False, id="reset-tail", marks=pytest.mark.smoke
        ),
        pytest.param(
            264,
            4096,
            torch.bfloat16,
            True,
            251,
            False,
            id="wide-reset-tail",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            257, 128, torch.float16, True, 251, True, id="tuned-reset-tail", marks=pytest.mark.full
        ),
    ],
)
def test_passing_scan(chunks, width, dtype, initial, period, tune):
    """Long scans and reset tails preserve previous and final states through the op."""
    test = PassingTest(1, chunks, 1, width, dtype, initial, reset_period=period)
    op = SSDStatePassingFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())
