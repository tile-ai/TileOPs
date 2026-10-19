"""Paged MLA decode: both schedules, both cache kinds, split runs and the cache's tail."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import TestBase
from tileops.backend import BUILTIN
from tileops.ops import MLAPagedFwdOp
from tileops.ops._signature_codegen import CheckError
from workloads.attention.mla import MLAPagedWorkload


class MLAPagedTest(MLAPagedWorkload, TestBase):
    """The workload as a test case: the reference and its policy come from the workload."""


BF16, FP16, FP8 = torch.bfloat16, torch.float16, torch.float8_e4m3fn


def _workload(
    heads, s_q, lens, page_size, dtype, cache_dtype=None, width=None, dim=576, dim_v=512, **kw
):
    width = width or max(-(-n // page_size) for n in lens)
    return MLAPagedTest(
        heads, s_q, dim, dim_v, lens, page_size, width, len(lens) * width + 3, dtype,
        cache_dtype, **kw,
    )  # fmt: skip


@pytest.mark.parametrize(
    "heads,s_q,lens,page_size,dtype,cache_dtype,width",
    [
        # Requests ending mid-tile and on a single key; the second tile of a pair missing.
        pytest.param(
            128, 1, [300, 64, 1], 64, BF16, BF16, None, marks=pytest.mark.smoke, id="bf16-ragged"
        ),
        # Two query tokens a request, each masked at its own position.
        pytest.param(
            64, 2, [129, 200], 64, FP16, FP16, None, marks=pytest.mark.smoke, id="fp16-mtp"
        ),
        pytest.param(64, 1, [700, 65], 64, BF16, FP8, None, marks=pytest.mark.smoke, id="fp8-bf16"),
        # FP8 tiles gathered across 16-row pages; 48 query rows fill part of a block.
        pytest.param(
            16, 3, [130, 3], 16, FP16, FP8, None, marks=pytest.mark.smoke, id="fp8-fp16-small-pages"
        ),
        # Pages of 7 rows take the gathering schedule.
        pytest.param(
            64, 1, [50, 13], 7, BF16, BF16, None, marks=pytest.mark.smoke, id="bf16-page-7"
        ),
        # Few blocks: the tiles are cut into runs. With a wide table one run of the short
        # request is empty, and one holds only the last key, which its first query
        # token may not see.
        pytest.param(
            128,
            2,
            [1000, 70, 129],
            64,
            BF16,
            BF16,
            16,
            marks=pytest.mark.smoke,
            id="bf16-split-runs",
        ),
        pytest.param(
            64, 1, [3000, 33], 64, BF16, FP8, 48, marks=pytest.mark.full, id="fp8-split-runs"
        ),
        # FP8 tiles on 7-row pages: page boundaries fall inside tiles.
        pytest.param(64, 1, [300, 22], 7, BF16, FP8, None, marks=pytest.mark.full, id="fp8-page-7"),
        # Enough blocks to fill the SMs: one run a request, up to 32 tiles long.
        pytest.param(
            128,
            1,
            [2000 - 37 * i for i in range(34)],
            64,
            BF16,
            BF16,
            None,
            marks=pytest.mark.full,
            id="bf16-long-runs",
        ),
    ],
)
def test_mla_paged_attends_each_request(heads, s_q, lens, page_size, dtype, cache_dtype, width):
    """Each query row attends to its request's visible keys; lse is their log-sum-exp."""
    workload = _workload(heads, s_q, lens, page_size, dtype, cache_dtype, width)
    workload.check(MLAPagedFwdOp(512), *workload.gen_inputs())


@pytest.mark.parametrize(
    "heads,s_q,lens,page_size,cache_dtype,is_causal,sm_scale,dim,dim_v",
    [
        pytest.param(
            32,
            2,
            [200, 65],
            64,
            BF16,
            False,
            None,
            576,
            512,
            marks=pytest.mark.smoke,
            id="non-causal",
        ),
        # A negative scale makes the smallest product the max; a zero one gives uniform
        # weights over the visible keys.
        pytest.param(
            64,
            1,
            [150, 64],
            64,
            BF16,
            True,
            -0.05,
            576,
            512,
            marks=pytest.mark.full,
            id="negative-scale",
        ),
        pytest.param(
            64,
            2,
            [150, 64],
            64,
            FP8,
            True,
            0.0,
            576,
            512,
            marks=pytest.mark.full,
            id="zero-scale",
        ),
        # A latent other than 512 + 64 takes the gathering schedule, FP8 included.
        pytest.param(
            32,
            1,
            [100, 40],
            16,
            BF16,
            True,
            None,
            192,
            128,
            marks=pytest.mark.full,
            id="latent-128",
        ),
        pytest.param(
            32,
            2,
            [100, 40],
            16,
            FP8,
            True,
            None,
            192,
            128,
            marks=pytest.mark.full,
            id="latent-128-fp8",
        ),
        # Widths off the 16-wide contraction step are padded with zero columns.
        pytest.param(
            16,
            2,
            [70, 9],
            16,
            BF16,
            True,
            None,
            40,
            8,
            marks=pytest.mark.full,
            id="value-8",
        ),
        pytest.param(
            16,
            1,
            [70, 9],
            7,
            FP8,
            True,
            None,
            20,
            10,
            marks=pytest.mark.full,
            id="value-10-rope-10-fp8",
        ),
        # A 16-wide value: its tile is padded to 32 columns for the two products.
        pytest.param(
            16,
            1,
            [70, 9],
            16,
            BF16,
            True,
            None,
            48,
            16,
            marks=pytest.mark.full,
            id="value-16",
        ),
    ],
)
def test_mla_paged_honors_the_parameters(
    heads, s_q, lens, page_size, cache_dtype, is_causal, sm_scale, dim, dim_v
):
    """Non-causal attention, a given score scale of any sign, and other latent widths."""
    workload = _workload(
        heads, s_q, lens, page_size, BF16, cache_dtype, dim=dim, dim_v=dim_v,
        is_causal=is_causal, sm_scale=sm_scale,
    )  # fmt: skip
    op = MLAPagedFwdOp(dim_v, is_causal=is_causal, sm_scale=sm_scale)
    workload.check(op, *workload.gen_inputs())


@pytest.mark.parametrize(
    "page_size,cache_dtype",
    [
        pytest.param(64, BF16, marks=pytest.mark.smoke, id="tma"),
        pytest.param(64, FP8, marks=pytest.mark.full, id="fp8"),
        pytest.param(7, BF16, marks=pytest.mark.full, id="page-7"),
    ],
)
def test_mla_paged_ignores_the_cache_past_each_request(page_size, cache_dtype):
    """A cache row past a request's length holding NaN reaches no result: a zero
    probability times NaN is NaN, so the value rows there are cleared."""
    workload = _workload(64, 1, [100, 65], page_size, BF16, cache_dtype)
    q, kv_cache, table, lens, kv_scale = workload.gen_inputs()
    for b, length in enumerate(workload.cache_lens):
        rows = kv_cache[table[b].long()].view(-1, kv_cache.shape[-1])
        rows[length:] = float("nan")
        kv_cache[table[b].long()] = rows.view(-1, page_size, kv_cache.shape[-1])
    workload.check(MLAPagedFwdOp(512), q, kv_cache, table, lens, kv_scale)


@pytest.mark.parametrize(
    "s_q,lens",
    [
        pytest.param(1, [390, 200], marks=pytest.mark.smoke, id="decode"),
        pytest.param(2, [320, 129], marks=pytest.mark.full, id="mtp"),
    ],
)
def test_mla_paged_follows_growing_scores(s_q, lens):
    """Rope columns four times larger each tile: a row's max grows past the stale-max
    allowance on tiles of both parities, and holding the old max would overflow the
    probabilities. The values keep their scale."""
    workload = _workload(64, s_q, lens, 64, BF16)
    q, kv_cache, table, lens_t, kv_scale = workload.gen_inputs()
    for b in range(len(lens)):
        for w in range(table.shape[1]):
            kv_cache[table[b, w].long(), :, 512:] *= 4.0**w
    workload.check(MLAPagedFwdOp(512), q, kv_cache, table, lens_t, kv_scale)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.parametrize(
    "lens,page_size,schedule,split",
    [
        # Three requests leave the SMs mostly idle: their tiles are cut into runs.
        pytest.param([300, 64, 1], 64, "MLAPagedSeesawKernel", True, id="seesaw-split"),
        # 34 requests of two 64-row blocks fill the SMs: one run a request.
        pytest.param([130] * 34, 64, "MLAPagedSeesawKernel", False, id="seesaw-whole"),
        pytest.param([50, 13], 7, "MLAPagedFwdKernel", False, id="gather"),
    ],
)
def test_mla_paged_takes_its_schedule_on_sm90(lens, page_size, schedule, split):
    """A 512 + 64 latent on pages of 64 rows takes the seesaw, other pages the gathering
    kernel; the seesaw cuts runs only while blocks are short of the SMs."""
    workload = _workload(128, 1, lens, page_size, BF16)
    inputs = workload.gen_inputs()
    op = MLAPagedFwdOp(512, target=BUILTIN)
    workload.check(op, *inputs)
    (kernel,) = op.built_entries("mla_paged").values()
    assert type(kernel).__name__ == schedule
    if schedule == "MLAPagedSeesawKernel":
        assert (kernel.num_splits(len(lens), inputs[2].shape[1]) > 1) == split


@pytest.mark.smoke
def test_mla_paged_rejects_dtypes_outside_the_signature():
    """A float32 query and a cache of the other 16-bit dtype fail the public call check."""
    workload = _workload(16, 1, [10], 16, BF16)
    q, kv_cache, table, lens, kv_scale = workload.gen_inputs()
    with pytest.raises(CheckError, match="dtype"):
        MLAPagedFwdOp(512)(q.float(), kv_cache, table, lens, kv_scale)
    with pytest.raises(CheckError, match="dtype"):
        MLAPagedFwdOp(512)(q, kv_cache.half(), table, lens, kv_scale)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_mla_paged_compiles_cold_to_its_operator():
    """A cold fullgraph compile owns the graph and returns what eager does."""
    workload = _workload(64, 1, [300, 65], 64, BF16)
    inputs = workload.gen_inputs()
    assert_op_owns_graph_nodes(MLAPagedFwdOp(512), *inputs)
    compiled = torch.compile(MLAPagedFwdOp(512), fullgraph=True)(*inputs)
    eager = MLAPagedFwdOp(512)(*inputs)
    torch.testing.assert_close(compiled[0], eager[0], atol=0, rtol=0)
    torch.testing.assert_close(compiled[1], eager[1], atol=0, rtol=0)


register_compile_contract(MLAPagedFwdOp)
