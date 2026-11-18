import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._utils import STR_TO_DTYPE
from tileops.attention import TopKSelectFwdOp
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
    op = TopKSelectFwdOp(top_k=topk)
    if tune:
        op.request_tune()
    inputs = test.gen_inputs()

    test.check(op, *inputs)


@pytest.mark.smoke
@pytest.mark.parametrize("fill_value", [None, -1], ids=["fill-default", "fill-minus-one"])
@pytest.mark.parametrize("width", [0, 5, 32], ids=["empty", "short", "exactly-topk"])
def test_topk_select_returns_a_short_window_whole(width: int, fill_value: int | None) -> None:
    """A window with at most ``topk`` keys selects all of them and pads with ``fill_value``."""
    batch, seq_len, seq_len_kv, topk = 2, 16, 256, 32
    scores = torch.randn(batch, seq_len, seq_len_kv, 1, device=run_device())
    starts = torch.full((batch, seq_len), 7, dtype=torch.int32, device=run_device())
    ends = starts + width
    out = TopKSelectFwdOp(top_k=topk, fill_value=fill_value)(scores, starts, ends)
    fill = seq_len_kv if fill_value is None else fill_value
    expected = sorted(list(range(7, 7 + width)) + [fill] * (topk - width))
    assert (out.sort(dim=-1).values == torch.tensor(expected, device=run_device())).all()
    workload = TopKSelectWorkload(
        batch, seq_len, seq_len_kv, 1, topk, torch.float32, torch.int32, fill_value
    )
    reference = workload.ref_program(scores, starts, ends)
    compare_outputs(out, reference, workload.verification(scores, starts, ends))


@pytest.mark.smoke
@pytest.mark.parametrize("fill_value", [None, -1], ids=["fill-default", "fill-minus-one"])
@pytest.mark.parametrize("seq_len_kv", [0, 1, 25, 511], ids=lambda n: f"keys{n}")
def test_topk_select_top_k_above_the_keys(seq_len_kv: int, fill_value: int | None) -> None:
    """Fewer keys than ``top_k``: each row selects its whole window and fills the rest.

    DeepSeek-V4-Flash's indexer takes the top 512 of a short context's compressed keys, none
    for a context under four tokens; each query sees a prefix of them.
    """
    batch, seq_len, kv_group, topk = 2, 16, 2, 512
    scores = torch.randn(batch, seq_len, seq_len_kv, kv_group, device=run_device())
    ends = torch.linspace(0, seq_len_kv, seq_len, device=run_device()).to(torch.int32)
    ends = ends.expand(batch, seq_len).contiguous()
    starts = torch.zeros_like(ends)
    starts[1] = ends[1] // 3
    out = TopKSelectFwdOp(top_k=topk, fill_value=fill_value)(scores, starts, ends)
    workload = TopKSelectWorkload(
        batch, seq_len, seq_len_kv, kv_group, topk, torch.float32, torch.int32, fill_value
    )
    compare_outputs(
        out, workload.ref_program(scores, starts, ends), workload.verification(scores, starts, ends)
    )


@pytest.mark.smoke
def test_topk_select_threshold_bucket_past_staging() -> None:
    """More keys share the topk-th score's bucket than the kernel stages; none is dropped."""
    seq_len_kv, topk = 8192, 100
    # Every score in [1, 1.03), one float16 bucket, and all distinct in float32.
    perm = torch.randperm(seq_len_kv, device=run_device()).float()
    scores = (1.0 + perm / (1 << 18)).reshape(1, 1, seq_len_kv, 1)
    starts = torch.zeros(1, 1, dtype=torch.int32, device=run_device())
    ends = torch.full((1, 1), seq_len_kv, dtype=torch.int32, device=run_device())
    out = TopKSelectFwdOp(top_k=topk)(scores, starts, ends)
    workload = TopKSelectWorkload(1, 1, seq_len_kv, 1, topk, torch.float32, torch.int32)
    compare_outputs(
        out, workload.ref_program(scores, starts, ends), workload.verification(scores, starts, ends)
    )


@pytest.mark.smoke
def test_topk_select_window_past_the_bin_cache() -> None:
    """A window past the keys whose bins stay in shared memory selects keys on both sides.

    The window starts and ends off a chunk boundary, past the first stage-1 chunk, and keys
    past the cached 65536 are binned again from global memory.
    """
    seq_len_kv, topk = 65536 + 8192, 256
    scores = torch.randn(1, 1, seq_len_kv, 1, device=run_device())
    # Half the winners inside the cached range, half past it; a stronger key before the
    # window must not be selected.
    scores[0, 0, 100, 0] += 16.0
    scores[0, 0, 10000 : 10000 + topk // 2, 0] += 8.0
    scores[0, 0, 70000 : 70000 + topk // 2, 0] += 8.0
    starts = torch.full((1, 1), 9000, dtype=torch.int32, device=run_device())
    ends = torch.full((1, 1), seq_len_kv - 3000, dtype=torch.int32, device=run_device())
    out = TopKSelectFwdOp(top_k=topk)(scores, starts, ends)
    workload = TopKSelectWorkload(1, 1, seq_len_kv, 1, topk, torch.float32, torch.int32)
    compare_outputs(
        out, workload.ref_program(scores, starts, ends), workload.verification(scores, starts, ends)
    )
