import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops import TopKSelectFwdOp
from tileops.utils import STR_TO_DTYPE
from workloads.attention.topk_select import TopKSelectWorkload
from workloads.device import run_device
from workloads.numerics import compare_outputs


class TopKSelectTest(TopKSelectWorkload, TestBase):
    pass


class TopKSelectFixture(FixtureBase):
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


@TopKSelectFixture
def test_topk_select_op(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    kv_group: int,
    topk: int,
    in_dtype_str: str,
    out_dtype_str: str,
    tune: bool,
) -> None:
    in_dtype = STR_TO_DTYPE[in_dtype_str]
    out_dtype = STR_TO_DTYPE[out_dtype_str]
    test = TopKSelectTest(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)
    op = TopKSelectFwdOp(topk=topk)
    if tune:
        op.autotune()
    inputs = test.gen_inputs()

    test.check(op, *inputs)


@pytest.mark.smoke
@pytest.mark.parametrize("width", [0, 5, 32], ids=["empty", "short", "exactly-topk"])
def test_topk_select_returns_a_short_window_whole(width: int) -> None:
    """A window with at most ``topk`` keys selects all of them and pads with ``seq_len_kv``."""
    batch, seq_len, seq_len_kv, topk = 2, 16, 256, 32
    scores = torch.randn(batch, seq_len, seq_len_kv, 1, device=run_device())
    starts = torch.full((batch, seq_len), 7, dtype=torch.int32, device=run_device())
    ends = starts + width
    out = TopKSelectFwdOp(topk=topk)(scores, starts, ends)
    expected = list(range(7, 7 + width)) + [seq_len_kv] * (topk - width)
    assert (out.sort(dim=-1).values == torch.tensor(expected, device=run_device())).all()
    workload = TopKSelectWorkload(batch, seq_len, seq_len_kv, 1, topk, torch.float32, torch.int32)
    reference = workload.ref_program(scores, starts, ends)
    compare_outputs(out, reference, workload.verification(scores, starts, ends))


@pytest.mark.smoke
def test_topk_select_threshold_bucket_past_staging() -> None:
    """More keys share the topk-th score's bucket than the kernel stages; none is dropped."""
    seq_len_kv, topk = 8192, 100
    # Every score in [1, 1.03), one float16 bucket, and all distinct in float32.
    perm = torch.randperm(seq_len_kv, device=run_device()).float()
    scores = (1.0 + perm / (1 << 18)).reshape(1, 1, seq_len_kv, 1)
    starts = torch.zeros(1, 1, dtype=torch.int32, device=run_device())
    ends = torch.full((1, 1), seq_len_kv, dtype=torch.int32, device=run_device())
    out = TopKSelectFwdOp(topk=topk)(scores, starts, ends)
    workload = TopKSelectWorkload(1, 1, seq_len_kv, 1, topk, torch.float32, torch.int32)
    compare_outputs(
        out, workload.ref_program(scores, starts, ends), workload.verification(scores, starts, ends)
    )
