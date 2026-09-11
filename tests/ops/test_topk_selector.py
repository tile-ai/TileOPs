import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import TopkSelectorFwdOp
from tileops.utils import str2dtype
from workloads.topk_selector import TopkSelectorWorkload


class TopkSelectorTest(TopkSelectorWorkload, TestBase):
    pass


class TopkSelectorFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, seq_len_kv, kv_group, topk, in_dtype_str, out_dtype_str, tune",
            [
                pytest.param(
                    4, 256, 1024, 1, 32, "float32", "int32", False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    8, 512, 2048, 1, 64, "float32", "int32", False, marks=pytest.mark.full
                ),
                pytest.param(
                    1,
                    32 * 1024,
                    64 * 1024,
                    1,
                    1024,
                    "float32",
                    "int32",
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1,
                    32 * 1024,
                    64 * 2048,
                    1,
                    2048,
                    "float32",
                    "int32",
                    False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@TopkSelectorFixture
def test_topk_selector_op(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    kv_group: int,
    topk: int,
    in_dtype_str: str,
    out_dtype_str: str,
    tune: bool,
) -> None:
    in_dtype = str2dtype[in_dtype_str]
    out_dtype = str2dtype[out_dtype_str]
    test = TopkSelectorTest(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)
    op = TopkSelectorFwdOp(topk=topk, tune=tune)
    inputs = test.gen_inputs()

    def compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
        def selected(indices: torch.Tensor) -> torch.Tensor:
            gather_index = indices.permute(0, 1, 3, 2).long()
            values = torch.gather(inputs[0], 2, gather_index).permute(0, 1, 3, 2)
            return torch.sort(values, dim=-1).values

        torch.testing.assert_close(selected(output), selected(output_ref))

    test.check(op, *inputs, compare=compare)
