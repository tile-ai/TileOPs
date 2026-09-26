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


@pytest.mark.smoke
@pytest.mark.parametrize("width", [0, 5, 32], ids=["empty", "short", "exactly-topk"])
def test_topk_selector_returns_a_short_window_whole(width: int) -> None:
    """A window with at most ``topk`` keys selects all of them and pads with ``seq_len_kv``."""
    batch, seq_len, seq_len_kv, topk = 2, 16, 256, 32
    scores = torch.randn(batch, seq_len, seq_len_kv, 1, device="cuda")
    starts = torch.full((batch, seq_len), 7, dtype=torch.int32, device="cuda")
    ends = starts + width
    out = TopkSelectorFwdOp(topk=topk)(scores, starts, ends)
    expected = list(range(7, 7 + width)) + [seq_len_kv] * (topk - width)
    assert (out.sort(dim=-1).values == torch.tensor(expected, device="cuda")).all()
