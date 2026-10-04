import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import FP8QuantFwdOp
from workloads.quantization.fp8_quant import FP8QuantWorkload


class FP8QuantTest(FP8QuantWorkload, TestBase):
    pass


class FP8QuantFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len_kv, kv_group, index_dim, dtype, tune",
            [
                pytest.param(1, 8192, 1, 64, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1, 8192, 1, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(1, 8192, 1, 64, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1, 4096, 1, 128, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1, 16384, 1, 32, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1, 1024, 4, 64, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@FP8QuantFixture
def test_fp8_quant_op(
    batch: int, seq_len_kv: int, kv_group: int, index_dim: int, dtype: torch.dtype, tune: bool
) -> None:
    test = FP8QuantTest(batch, seq_len_kv, kv_group, index_dim, dtype)
    op = FP8QuantFwdOp(tune=tune)
    test.check(op, *test.gen_inputs())
