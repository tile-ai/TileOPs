from typing import Tuple

import torch
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops import TopkSelectorOp
from tileops.utils import str2dtype


class TopkSelectorFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, topk, in_dtype_str, out_dtype_str, tune", [
            (64, 32 * 1024, 1024, "float32", "int32", False),
            (64, 32 * 1024, 2048, "float32", "int32", False),
            (128, 64 * 1024, 1024, "float32", "int32", False),
            (128, 64 * 1024, 2048, "float32", "int32", False),
        ]),
    ]


def _set_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using set intersection (topk indices may be in different order)."""
    ref_np = output_ref.cpu().to(torch.int32).numpy()
    trt_np = output.cpu().to(torch.int32).numpy()

    set_ref = set(ref_np.flatten().tolist())
    set_trt = set(trt_np.flatten().tolist())
    intersection = set_ref & set_trt
    assert len(intersection) / len(set_ref) == 1.0, \
        "output indices do not match reference indices"


class TopkSelectorTest(TestBase):

    def __init__(self, batch: int, seq_len: int, topk: int, in_dtype: torch.dtype,
                 out_dtype: torch.dtype):
        self.batch = batch
        self.seq_len = seq_len
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index_score = torch.randn(self.batch, self.seq_len, dtype=self.in_dtype, device="cuda")
        starts = torch.zeros(self.batch, dtype=self.out_dtype, device="cuda")
        ends = torch.ones(self.batch, dtype=self.out_dtype, device="cuda") * self.seq_len
        return index_score, starts, ends

    def ref_program(self, index_score: torch.Tensor, starts: torch.Tensor,
                    ends: torch.Tensor) -> torch.Tensor:
        indexes_ref = torch.topk(index_score, self.topk, dim=-1)[1]
        return indexes_ref


@TopkSelectorFixture
def test_topk_selector_op(batch: int, seq_len: int, topk: int, in_dtype_str: str,
                          out_dtype_str: str, tune: bool) -> None:
    in_dtype = str2dtype[in_dtype_str]
    out_dtype = str2dtype[out_dtype_str]
    test = TopkSelectorTest(batch, seq_len, topk, in_dtype, out_dtype)
    op = TopkSelectorOp(batch, seq_len, topk, in_dtype, out_dtype, tune=tune)
    test.check(op, *test.gen_inputs(), compare=_set_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
