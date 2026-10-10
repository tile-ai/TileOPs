"""GQAPagedFwdOp tests against the workload's reference."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops import GQAPagedFwdOp
from workloads.attention.gqa.paged import GQAPagedFwdWorkload


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
) -> GQAPagedFwdWorkload:
    """A decode call: ``q_len`` query tokens per request, tables as wide as the longest cache."""
    width = -(-max(cache_lens) // page_size)
    return GQAPagedFwdWorkload(
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
    TestBase.check(workload, op, *inputs)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("num_split", [1, 4], ids=["unsplit", "split"])
@pytest.mark.parametrize("pos_encoding_mode", ["none", "rope"])
def test_gqa_paged_negative_scale(num_split: int, pos_encoding_mode: str) -> None:
    """Negative scales must preserve masks and finite online softmax state and split LSE."""
    from tileops.kernels.attention.gqa.paged import GQAPagedFwdKernel

    class FixedSplitKernel(GQAPagedFwdKernel):
        @property
        def default_config(self):
            return {**super().default_config, "num_split": num_split}

    semantics = dict(sm_scale=-0.125, pos_encoding_mode=pos_encoding_mode)
    workload = GQAPagedFwdWorkload(
        16, 4, 128, [3, 1], [257, 33], 64, 5, 10, torch.float16, **semantics
    )
    op = GQAPagedFwdOp(**semantics, kernel_map={"gqa_paged_varlen_kernel": FixedSplitKernel})
    _check(op, workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "q_len,cache_len", [(1, 1025), (4, 4097)], ids=["single-token", "causal-split"]
)
def test_gqa_paged_decode_negative_scale(q_len: int, cache_len: int) -> None:
    """Uniform decode must not turn mask sentinels into positive infinity."""
    from tileops.kernels.attention.gqa.paged_decode import GQADecodePagedKernel

    workload = _decode(1, 16, 4, [cache_len], 128, 128, q_len=q_len, sm_scale=-0.125)
    # Keep single-token dispatch natural to guard the batch-1 specialization's refusal.
    kernel_map = {"gqa_paged_varlen_kernel": GQADecodePagedKernel} if q_len > 1 else None
    _check(GQAPagedFwdOp(sm_scale=-0.125, kernel_map=kernel_map), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype,q_lens,cache_lens,page_size,dim,semantics",
    [
        pytest.param(torch.float16, [1, 3], [129, 63], 64, 128, {}, id="fp16-split"),
        pytest.param(torch.bfloat16, [1, 3], [129, 63], 64, 128, {}, id="bf16-split"),
        pytest.param(
            torch.float16,
            [160, 129],
            [256, 193],
            256,
            64,
            {"rope_layout": "interleaved"},
            id="interleaved-unsplit",
        ),
        pytest.param(
            torch.float16,
            [7, 0, 130],
            [99, 0, 145],
            48,
            128,
            {"rotary_dim": 48, "sm_scale": 0.25},
            id="partial-odd-pages",
        ),
        pytest.param(
            torch.float16,
            [2, 1],
            [65, 97],
            16,
            256,
            {"rotary_dim": 64, "rope_layout": "interleaved", "softcap": 2.0},
            id="partial-interleaved-softcap",
        ),
        pytest.param(
            torch.float16,
            [65, 3],
            [129, 33],
            64,
            128,
            {"is_causal": False, "window_size_left": 17, "window_size_right": 3},
            id="two-sided-window",
        ),
        pytest.param(
            torch.float16,
            [3, 1],
            [65, 33],
            64,
            128,
            {"is_causal": False, "sm_scale": 0.0},
            id="noncausal-zero-scale",
        ),
    ],
)
def test_gqa_paged_rope_reads_logical_positions(
    dtype: torch.dtype,
    q_lens: list[int],
    cache_lens: list[int],
    page_size: int,
    dim: int,
    semantics: dict,
) -> None:
    """Rotate logical positions through fragmented pages without modifying the cache.

    Poisoned cache tails must not enter either the rotary table lookup or the value sum.
    """
    semantics = {"pos_encoding_mode": "rope", **semantics}
    width = -(-max(cache_lens) // page_size)
    workload = GQAPagedFwdWorkload(
        16,
        4,
        dim,
        q_lens,
        cache_lens,
        page_size,
        width,
        len(q_lens) * width,
        dtype,
        **semantics,
    )
    inputs = workload.gen_inputs()
    for request, length in enumerate(cache_lens):
        slots = torch.arange(length, width * page_size, device=inputs[0].device)
        stale = (inputs[3][request, slots // page_size], slots % page_size)
        inputs[1][stale] = float("nan")
        inputs[2][stale] = float("inf")
    _check(GQAPagedFwdOp(**semantics), workload, inputs)


@pytest.mark.smoke
@pytest.mark.cuda_only
def test_gqa_paged_rope_replays_positions_from_device_lengths() -> None:
    """Replay must rotate at the current cache positions and current packed Q boundaries."""
    workload = GQAPagedFwdWorkload(
        16,
        4,
        128,
        [1, 3],
        [129, 65],
        64,
        3,
        6,
        torch.float16,
        pos_encoding_mode="rope",
        rotary_dim=64,
    )
    inputs = workload.gen_inputs()
    op = GQAPagedFwdOp(pos_encoding_mode="rope", rotary_dim=64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    inputs[4].copy_(torch.tensor([33, 63], device=inputs[0].device, dtype=torch.int32))
    inputs[5][1] = 2
    workload.q_lens, workload.cache_lens = [2, 2], [33, 63]
    graph.replay()
    TestBase.check(workload, op, *inputs, runs=lambda *args: output)


class GQAPagedDecodeFixture(FixtureBase):
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


@GQAPagedDecodeFixture
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
    _check(GQAPagedFwdOp(), workload, workload.gen_inputs())


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
    op = GQAPagedFwdOp(sm_scale=sm_scale, softcap=softcap)
    _check(op, workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("batch", [pytest.param(2, id="generic"), pytest.param(1, id="bs1")])
def test_gqa_paged_decode_pool_is_independent_of_table_width(batch: int) -> None:
    """A pool holding more pages than the tables name reads only the named ones."""
    workload = _decode(batch, 32, 4, [2048] * batch, 128, 256, pool_pages=4 * batch * 8)
    _check(GQAPagedFwdOp(), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("page_size", [192, 96, 48])
def test_gqa_paged_reversed_page_table(page_size: int) -> None:
    """A page table that permutes the pool is read through, for pages of any length."""
    workload = _decode(1, 16, 4, [page_size * 16], 128, page_size)
    inputs = list(workload.gen_inputs())
    inputs[3] = inputs[3].flip(-1).contiguous()
    _check(GQAPagedFwdOp(), workload, inputs)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "cache_lens",
    [pytest.param([700, 130], id="no-split"), pytest.param([2048, 1500], id="split")],
)
def test_gqa_paged_multi_token_causal(cache_lens: list[int]) -> None:
    """Several query tokens per request, with a head group's rows spanning two row blocks."""
    workload = _decode(2, 32, 2, cache_lens, 64, 64, q_len=5)
    _check(GQAPagedFwdOp(), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("cache_lens", "page_size", "q_len"),
    [
        # Pages of 64 hold copied key tiles of 64 rows, and pages of 256 tiles of 128. Five
        # query tokens a request leave the device short of row tiles, so the key range is
        # split; 160 fill it, so it is not.
        pytest.param([63, 130], 64, 5, id="tile-64-split"),
        pytest.param([2047, 1501], 256, 5, id="tile-128-split"),
        pytest.param([700, 450], 64, 160, id="tile-64"),
        pytest.param([700, 450], 256, 160, id="tile-128"),
    ],
)
def test_gqa_paged_drops_stale_rows_past_the_cache(
    cache_lens: list[int], page_size: int, q_len: int
) -> None:
    """A page keeps stale rows past a request's cache; a NaN among them must not reach the
    output through the key tile the cache end cuts."""
    workload = _decode(2, 32, 8, cache_lens, 128, page_size, q_len=q_len)
    inputs = workload.gen_inputs()
    k_pages, v_pages, page_table = inputs[1], inputs[2], inputs[3]
    for request, length in enumerate(cache_lens):
        slots = torch.arange(length, page_table.shape[1] * page_size, device=v_pages.device)
        stale = (page_table[request, slots // page_size], slots % page_size)
        k_pages[stale] = float("nan")
        v_pages[stale] = float("nan")
    _check(GQAPagedFwdOp(), workload, inputs)


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
    op = GQAPagedFwdOp()
    _check(op, workload, inputs)


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
    workload = GQAPagedFwdWorkload(
        32, 8, dim, q_lens, cache_lens, page_size, width, len(q_lens) * width, dtype, **op_kwargs
    )
    _check(GQAPagedFwdOp(**op_kwargs), workload, workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize("batch", [2, 1], ids=["dynamic", "single-request"])
def test_gqa_paged_graph_replays_device_metadata(batch: int) -> None:
    """Dispatch is static; packed lengths and cache lengths remain replay inputs."""
    workload = _decode(batch, 32, 4, [4096, 64][:batch], 128, 64)
    inputs = workload.gen_inputs()
    inputs[4][0] = 128  # A mostly empty table must also be safe on the split path.
    workload.cache_lens[0] = 128
    op = GQAPagedFwdOp()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    # Same tensor addresses and shapes, different sequence boundaries. The dynamic
    # path must not keep the uniform packing it saw during warmup or capture.
    if batch == 2:
        inputs[5][1] = 0
        workload.q_lens = [0, 2]
        inputs[4][-1] = 63
        workload.cache_lens[-1] = 63
    v_pages, page_table, page_size = inputs[2], inputs[3][0], inputs[2].shape[1]
    live_values = v_pages.clone()
    for length in (128, 2048, 63, 0):
        inputs[4][0] = length
        workload.cache_lens[0] = length
        # Slots past the cache keep stale contents; a non-finite one must not reach the output.
        v_pages.copy_(live_values)
        slots = torch.arange(length, page_table.numel() * page_size, device=v_pages.device)
        v_pages[page_table[slots // page_size], slots % page_size] = float("nan")
        graph.replay()
        TestBase.check(workload, op, *inputs, runs=lambda *args: output)
