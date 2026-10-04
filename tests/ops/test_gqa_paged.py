"""GroupedQueryAttentionPagedFwdOp tests against the workload's reference."""

import pytest
import torch

from tests.test_base import FixtureBase
from tileops.ops import GroupedQueryAttentionPagedFwdOp
from workloads.attention.gqa.paged import GroupedQueryAttentionPagedFwdWorkload
from workloads.numerics import reference_tolerance


def _decode(
    batch: int,
    heads: int,
    heads_kv: int,
    cache_lens: list[int],
    dim: int,
    page_size: int,
    dtype: torch.dtype = torch.float16,
    *,
    pool_pages: int | None = None,
    q_len: int = 1,
    **semantics,
) -> GroupedQueryAttentionPagedFwdWorkload:
    """A decode call: ``q_len`` query tokens per request, tables as wide as the longest cache."""
    width = -(-max(cache_lens) // page_size)
    return GroupedQueryAttentionPagedFwdWorkload(
        heads,
        heads_kv,
        dim,
        [q_len] * batch,
        cache_lens,
        page_size,
        width,
        pool_pages or batch * width,
        dtype,
        **semantics,
    )


def _check(op, workload, inputs) -> None:
    torch.testing.assert_close(
        op(*inputs), workload.ref_program(*inputs), **reference_tolerance(workload.dtype)
    )


def _built_kernel(op, inputs):
    """The kernel *op* selects and builds for *inputs*."""
    q, k_pages, _, page_table, _, cu_seqlens_q = inputs[:6]
    call = op.paged_call(q, k_pages, page_table, cu_seqlens_q)
    return op.kernel_for("gqa_paged", call)


class GroupedQueryAttentionPagedDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, heads_kv, cache_lens, dim, page_size, dtype",
            [
                pytest.param(
                    2, 16, 4, [513, 1000], 128, 128, torch.float16, marks=pytest.mark.smoke
                ),
                pytest.param(
                    2, 16, 4, [513, 1000], 128, 128, torch.bfloat16, marks=pytest.mark.smoke
                ),
                # The default tile at head dim 512 overflows every GPU's shared memory.
                pytest.param(
                    2, 16, 4, [513, 1000], 512, 128, torch.float16, marks=pytest.mark.smoke
                ),
                pytest.param(1, 16, 8, [512], 128, 128, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 8, 4, [1024, 700], 64, 256, torch.float16, marks=pytest.mark.full),
                pytest.param(1, 32, 8, [200], 128, 64, torch.float16, marks=pytest.mark.full),
                pytest.param(1, 16, 4, [2048], 128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(1, 32, 16, [512], 64, 128, torch.float16, marks=pytest.mark.full),
                pytest.param(
                    4, 32, 8, [100, 16, 77, 300], 128, 16, torch.float16, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@GroupedQueryAttentionPagedDecodeFixture
def test_gqa_paged_decode_op(
    batch: int,
    heads: int,
    heads_kv: int,
    cache_lens: list[int],
    dim: int,
    page_size: int,
    dtype: torch.dtype,
) -> None:
    workload = _decode(batch, heads, heads_kv, cache_lens, dim, page_size, dtype)
    _check(GroupedQueryAttentionPagedFwdOp(), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "sm_scale, softcap, cache_lens, q_len",
    [
        pytest.param(0.25, None, [512, 300], 1, id="custom-sm-scale"),
        pytest.param(None, 2.0, [512, 300], 1, id="softcap"),
        # Unequal lengths on the split path leave the short request with empty splits.
        pytest.param(0.0, None, [4096, 128], 1, id="zero-scale-split"),
        # Masked keys stay masked when every score is zero.
        pytest.param(0.0, None, [700, 300], 4, id="zero-scale-causal"),
    ],
)
def test_gqa_paged_decode_score_controls(
    sm_scale: float | None, softcap: float | None, cache_lens: list[int], q_len: int
) -> None:
    workload = _decode(
        2, 16, 8, cache_lens, 128, 128, q_len=q_len, sm_scale=sm_scale, softcap=softcap
    )
    op = GroupedQueryAttentionPagedFwdOp(sm_scale=sm_scale, softcap=softcap)
    _check(op, workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("batch", [pytest.param(2, id="generic"), pytest.param(1, id="bs1")])
def test_gqa_paged_decode_pool_is_independent_of_table_width(batch: int) -> None:
    """A pool holding more pages than the tables name reads only the named ones."""
    workload = _decode(batch, 32, 4, [2048] * batch, 128, 256, pool_pages=4 * batch * 8)
    _check(GroupedQueryAttentionPagedFwdOp(), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("page_size", [192, 96, 48])
def test_gqa_paged_reversed_page_table(page_size: int) -> None:
    """A page table that permutes the pool is read through, for pages of any length."""
    workload = _decode(1, 16, 4, [page_size * 16], 128, page_size)
    inputs = list(workload.gen_inputs())
    inputs[3] = inputs[3].flip(-1).contiguous()
    _check(GroupedQueryAttentionPagedFwdOp(), workload, inputs)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "cache_lens",
    [pytest.param([700, 130], id="no-split"), pytest.param([2048, 1500], id="split")],
)
def test_gqa_paged_multi_token_causal(cache_lens: list[int]) -> None:
    """Several query tokens per request, with a head group's rows spanning two row blocks."""
    workload = _decode(2, 32, 2, cache_lens, 64, 64, q_len=5)
    _check(GroupedQueryAttentionPagedFwdOp(), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("cache_len", "reverse_pages"),
    [
        pytest.param(512, False, id="no-split"),
        pytest.param(4096, True, id="ctx-reversed-pages"),
    ],
)
def test_gqa_paged_decode_bs1_tiers(cache_len: int, reverse_pages: bool) -> None:
    """Both batch-1 tiers, including a page translation that changes the output."""
    torch.manual_seed(0)
    workload = _decode(1, 32, 4, [cache_len], 128, 256)
    inputs = list(workload.gen_inputs())
    if reverse_pages:
        inputs[3] = inputs[3].flip(-1).contiguous()
    op = GroupedQueryAttentionPagedFwdOp()
    kernel = _built_kernel(op, inputs)
    assert kernel.__class__.__name__ == "GQADecodePagedBs1Kernel"
    assert kernel._select_tier(cache_len) == ("ctx" if cache_len >= 1024 else "no_split")
    _check(op, workload, inputs)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
def test_gqa_paged_decode_bs1_dispatch() -> None:
    """An eligible batch-1 call selects the batch-1 kernel and its tiers."""
    workload = _decode(1, 32, 4, [8192], 128, 256)
    kernel = _built_kernel(GroupedQueryAttentionPagedFwdOp(), workload.gen_inputs())
    assert kernel.__class__.__name__ == "GQADecodePagedBs1Kernel"
    assert kernel._select_tier(1024) == "ctx"
    assert kernel._select_tier(512) == "no_split"
    assert kernel._ctx_splits_for(8192) == 32
    assert kernel._ctx_splits_for(2048) == 16
    assert kernel._ctx_splits_for(3072) == 8


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("q_lens", "cache_lens", "page_size", "dim", "dtype", "op_kwargs"),
    [
        pytest.param(
            [0, 1, 5, 1], [300, 512, 600, 7], 64, 128, torch.float16, {}, id="ragged-with-empty"
        ),
        # Head dimension 64, where the key tile is gathered at a width the page does not hold.
        pytest.param(
            [1, 1], [1000, 513], 64, 64, torch.float16, {"window_size_left": 128}, id="window"
        ),
        pytest.param(
            [7, 3],
            [300, 200],
            128,
            128,
            torch.float16,
            {"is_causal": False, "window_size_left": 64, "window_size_right": 32},
            id="both-windows-noncausal",
        ),
        # A page no key tile fits inside: the key tile is gathered row by row.
        pytest.param([1, 1], [400, 131], 65, 128, torch.float16, {}, id="page-no-tile-fits"),
        # The only shape where the causal bound and the window bound both bind: a request
        # with one query token sees its whole cache, so its causal bound masks nothing.
        pytest.param(
            [6, 1, 0],
            [300, 500, 128],
            64,
            128,
            torch.float16,
            {"window_size_left": 64},
            id="causal-window-multi",
        ),
        pytest.param(
            [2, 0, 9], [700, 64, 33], 16, 128, torch.float16, {"softcap": 30.0}, id="softcap"
        ),
        pytest.param([3, 1], [129, 48], 48, 128, torch.bfloat16, {}, id="bf16"),
        # The three shapes that isolate one term of the key-tile classification each. A tile
        # is masked per element only where a bound cuts it, so a term that never fires on any
        # case is a term no test pays for.
        # Causal only: a cache length the key tile divides, so the cache end never cuts and
        # the causal bound is the only term that can.
        pytest.param([5, 2], [512, 256], 64, 128, torch.float16, {}, id="causal-aligned-cache"),
        # Neither causal nor windowed: the cache end is then the only term that can cut.
        pytest.param(
            [3, 2], [250, 130], 64, 128, torch.float16, {"is_causal": False}, id="cache-end-only"
        ),
        # A right window narrower than the request's own query span, over a cache the key
        # tile divides: the right bound is then the only term that cuts.
        pytest.param(
            [64, 8],
            [256, 128],
            64,
            128,
            torch.float16,
            {"is_causal": False, "window_size_right": 8},
            id="right-window-only",
        ),
    ],
)
def test_gqa_paged_packed_query_lengths(
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dim: int,
    dtype: torch.dtype,
    op_kwargs: dict,
) -> None:
    """Calls the decode region does not serve: ragged lengths, windows, and odd pages."""
    width = -(-max(cache_lens) // page_size)
    workload = GroupedQueryAttentionPagedFwdWorkload(
        32, 8, dim, q_lens, cache_lens, page_size, width, len(q_lens) * width, dtype, **op_kwargs
    )
    _check(GroupedQueryAttentionPagedFwdOp(**op_kwargs), workload, workload.gen_inputs())
