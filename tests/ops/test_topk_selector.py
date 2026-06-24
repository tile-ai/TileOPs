
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import TopkSelectorOp
from tileops.utils import str2dtype
from workloads.topk_selector import TopkSelectorTest as _TopkSelectorTestWorkload


class TopkSelectorTest(_TopkSelectorTestWorkload, TestBase):
    def ref_program(self, index_score: torch.Tensor, starts: torch.Tensor,
                    ends: torch.Tensor) -> torch.Tensor:
        # index_score: (batch, seq_len, seq_len_kv, kv_group); topk over seq_len_kv (dim=2)
        indexes_ref = torch.topk(index_score, self.topk, dim=2)[1]
        # Match kernel/output layout: (batch, seq_len, kv_group, topk)
        return indexes_ref.permute(0, 1, 3, 2)


class TopkSelectorFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, seq_len_kv, kv_group, topk, in_dtype_str, out_dtype_str, tune", [
            pytest.param(4, 256, 1024, 1, 32, "float32", "int32", False, marks=pytest.mark.smoke),
            pytest.param(8, 512, 2048, 1, 64, "float32", "int32", False, marks=pytest.mark.full),
            pytest.param(1, 32 * 1024, 64 * 1024, 1, 1024, "float32", "int32", False, marks=pytest.mark.full),
            pytest.param(1, 32 * 1024, 64 * 2048, 1, 2048, "float32", "int32", False, marks=pytest.mark.full),
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


@TopkSelectorFixture
def test_topk_selector_op(batch: int,
                          seq_len: int,
                          seq_len_kv: int,
                          kv_group: int,
                          topk: int,
                          in_dtype_str: str,
                          out_dtype_str: str,
                          tune: bool) -> None:
    in_dtype = str2dtype[in_dtype_str]
    out_dtype = str2dtype[out_dtype_str]
    test = TopkSelectorTest(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)
    op = TopkSelectorOp(batch=batch,
                        seq_len=seq_len,
                        seq_len_kv=seq_len_kv,
                        kv_group=kv_group,
                        topk=topk,
                        in_dtype=in_dtype,
                        out_dtype=out_dtype,
                        tune=tune)
    test.check(op, *test.gen_inputs(), compare=_set_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
