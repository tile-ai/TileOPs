"""Public paged KV cache write semantics, both cache dtypes and the skip path."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.cuda_graph_contract import check_replay
from tests.workload_test_base import TestBase
from tileops.attention import PagedKVCacheWriteFwdOp
from workloads.attention.paged_cache_write import (
    PagedKVCacheWriteWorkload,
    paged_cache_write_result,
)


class PagedKVCacheWriteTest(PagedKVCacheWriteWorkload, TestBase):
    """The workload as a test case: the reference and its policy come from the workload."""


@pytest.mark.parametrize(
    "tokens,heads_kv,dim,page_size,num_pages,dtype,cache_dtype,slots,tune",
    [
        pytest.param(
            5,
            2,
            64,
            7,
            3,
            torch.float16,
            torch.float16,
            [20, -1, 0, 6, 13],
            False,
            marks=pytest.mark.smoke,
            id="fp16-odd-page-skip-and-page-edges",
        ),
        pytest.param(
            4,
            1,
            4,
            4,
            2,
            torch.bfloat16,
            torch.bfloat16,
            [7, 0, 3, 4],
            False,
            marks=pytest.mark.smoke,
            id="bf16-one-kv-head-last-row",
        ),
        pytest.param(
            3,
            2,
            64,
            16,
            2,
            torch.bfloat16,
            torch.float8_e4m3fn,
            [1, -1, 31],
            False,
            marks=pytest.mark.smoke,
            id="fp8-bf16-scaled",
        ),
        pytest.param(
            2,
            1,
            5,
            16,
            4,
            torch.float16,
            torch.float8_e4m3fn,
            [63, 0],
            False,
            marks=pytest.mark.full,
            id="fp8-fp16-page-boundary",
        ),
        pytest.param(
            64,
            8,
            128,
            64,
            8,
            torch.bfloat16,
            torch.bfloat16,
            None,
            False,
            marks=pytest.mark.full,
            id="bf16-decode-step-distinct-slots",
        ),
        pytest.param(
            4096,
            8,
            128,
            16,
            256,
            torch.float16,
            torch.float16,
            None,
            False,
            marks=pytest.mark.full,
            id="fp16-prefill-sweep",
        ),
        pytest.param(
            5,
            3,
            4,
            4,
            2,
            torch.float16,
            torch.float8_e4m3fn,
            [0, -1, 7, 2, 5],
            False,
            marks=pytest.mark.full,
            id="fp8-eight-byte-load-four-byte-store",
        ),
        pytest.param(
            5,
            1,
            3,
            4,
            2,
            torch.float16,
            torch.float16,
            [7, -1, 0, 4, 3],
            False,
            marks=pytest.mark.full,
            id="fp16-odd-row-element-copy",
        ),
        pytest.param(
            5,
            1,
            6,
            4,
            2,
            torch.bfloat16,
            torch.float8_e4m3fn,
            [6, 0, -1, 3, 7],
            False,
            marks=pytest.mark.full,
            id="fp8-two-byte-vector-branched-store",
        ),
        pytest.param(
            6,
            2,
            3,
            7,
            3,
            torch.bfloat16,
            torch.bfloat16,
            [4, 20, 1, -1, 13, 6],
            True,
            marks=pytest.mark.full,
            id="bf16-tuned",
        ),
    ],
)
def test_paged_kv_cache_write_scatters_the_named_rows(
    tokens, heads_kv, dim, page_size, num_pages, dtype, cache_dtype, slots, tune
):
    """Every row a slot names holds the new k and v; the rest of both pools is untouched."""
    workload = PagedKVCacheWriteTest(
        tokens, heads_kv, dim, page_size, num_pages, dtype, cache_dtype, slots
    )
    inputs = workload.gen_inputs()
    op = PagedKVCacheWriteFwdOp()
    if tune:
        op.request_tune()
    TestBase.check(workload, op, *inputs, runs=lambda *args: paged_cache_write_result(op, *args))


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="PagedKVCacheWriteFwdOp")
@pytest.mark.parametrize(
    "num_pages",
    [
        pytest.param(64, marks=pytest.mark.smoke, id="qwen3.8-27b"),
        pytest.param(65, marks=pytest.mark.full, id="qwen3.8-27b-4128"),
    ],
)
def test_paged_kv_cache_write_graph_replay(num_pages: int) -> None:
    """Qwen3.8-27B's decode call, one bfloat16 token into the pages of max_seq_len 4096 or 4128.
    Every draw repeats its rows, caches and slot, so the replay negates the values and writes
    the next row, the first of the last page."""
    workload = PagedKVCacheWriteTest(
        1, 4, 256, 64, num_pages, torch.bfloat16, slots=[(num_pages - 1) * 64 - 1]
    )
    key, value, k_cache, v_cache, slots, *rest = workload.gen_inputs()
    inputs = (-key, -value, -k_cache, -v_cache, slots + 1, *rest)
    check_replay(workload, PagedKVCacheWriteFwdOp(), inputs, result=paged_cache_write_result)


@pytest.mark.smoke
def test_paged_kv_cache_write_saturates_fp8_overflow():
    """FP8 preserves NaN, saturates overflow/infinities, and rounds ties to even."""
    workload = PagedKVCacheWriteTest(1, 1, 16, 2, 1, torch.float16, torch.float8_e4m3fn, [1])
    inputs = workload.gen_inputs()
    k, v, _, _, _, k_scale, v_scale = inputs
    values = torch.tensor(
        [
            1000,
            -1000,
            448,
            -448,
            1.2,
            -1.2,
            1.0625,
            1.1875,
            float("nan"),
            float("inf"),
            -float("inf"),
            0,
            -0.0,
            1,
            -1,
            0.5,
        ],
        dtype=k.dtype,
        device=k.device,
    )
    k.copy_(values.reshape_as(k))
    v.copy_(-values.reshape_as(v))
    k_scale.fill_(1)
    v_scale.fill_(1)
    op = PagedKVCacheWriteFwdOp()
    TestBase.check(workload, op, *inputs, runs=lambda *args: paged_cache_write_result(op, *args))


@pytest.mark.smoke
def test_paged_kv_cache_write_skips_every_token():
    """All-negative slots leave both sentinel-filled page pools unchanged."""
    workload = PagedKVCacheWriteTest(3, 2, 64, 7, 4, torch.bfloat16, slots=[-1, -1, -1])
    k, v, k_cache, v_cache, slots, k_scale, v_scale = workload.gen_inputs()
    k_cache.fill_(3.0)
    v_cache.fill_(-2.0)
    before = (k_cache.clone(), v_cache.clone())
    op = PagedKVCacheWriteFwdOp()
    assert op(k, v, k_cache, v_cache, slots, k_scale, v_scale) is None
    assert torch.equal(k_cache, before[0]) and torch.equal(v_cache, before[1])
    TestBase.check(
        workload,
        op,
        k,
        v,
        before[0],
        before[1],
        slots,
        k_scale,
        v_scale,
        runs=lambda *args: paged_cache_write_result(op, *args),
    )


@pytest.mark.smoke
def test_paged_kv_cache_write_past_the_pool_writes_nothing():
    """A slot of ``num_pages * page_size`` writes neither pool nor the page after it."""
    num_pages, page_size = 2, 16
    workload = PagedKVCacheWriteTest(1, 2, 64, page_size, num_pages, torch.bfloat16)
    k, v, k_cache, v_cache, _, k_scale, v_scale = workload.gen_inputs()
    # Each pool is a view of a buffer one page longer, so a store past the pool lands in
    # memory the test owns and can see.
    k_buffer = torch.full(
        (num_pages + 1, *k_cache.shape[1:]), 3.0, dtype=k_cache.dtype, device=k_cache.device
    )
    v_buffer = torch.full_like(k_buffer, -2.0)
    before = (k_buffer.clone(), v_buffer.clone())
    slots = torch.tensor([num_pages * page_size], dtype=torch.int64, device=k.device)
    op = PagedKVCacheWriteFwdOp()
    op(k, v, k_buffer[:num_pages], v_buffer[:num_pages], slots, k_scale, v_scale)
    torch.cuda.synchronize()
    assert torch.equal(k_buffer, before[0]) and torch.equal(v_buffer, before[1])


@pytest.mark.smoke
def test_paged_kv_cache_write_of_no_tokens_writes_nothing():
    """A call with no token writes neither pool; the caches it is handed are non-empty."""
    workload = PagedKVCacheWriteTest(0, 2, 64, 16, 2, torch.bfloat16)
    k, v, k_cache, v_cache, slots, k_scale, v_scale = workload.gen_inputs()
    before = (k_cache.clone(), v_cache.clone())
    op = PagedKVCacheWriteFwdOp()
    assert op(k, v, k_cache, v_cache, slots, k_scale, v_scale) is None
    assert torch.equal(k_cache, before[0]) and torch.equal(v_cache, before[1])


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("scale-missing", marks=pytest.mark.smoke),
        pytest.param("scale-present", marks=pytest.mark.smoke),
        pytest.param("input-dtype", marks=pytest.mark.full),
        pytest.param("cache-dtype", marks=pytest.mark.full),
    ],
)
def test_paged_kv_cache_write_refuses_an_invalid_tensor_contract(invalid):
    """Unsupported input/cache dtypes and incompatible scale presence are rejected."""
    cache_dtype = torch.float8_e4m3fn if invalid == "scale-missing" else torch.bfloat16
    workload = PagedKVCacheWriteTest(2, 2, 64, 16, 2, torch.bfloat16, cache_dtype)
    inputs = list(workload.gen_inputs())
    if invalid == "scale-missing":
        inputs[5] = inputs[6] = None
    elif invalid == "scale-present":
        inputs[5] = torch.ones(1, dtype=torch.float32, device=inputs[0].device)
        inputs[6] = inputs[5].clone()
    elif invalid == "input-dtype":
        inputs[0], inputs[1] = inputs[0].float(), inputs[1].float()
    else:
        inputs[2], inputs[3] = inputs[2].float(), inputs[3].float()
    with pytest.raises(ValueError, match="dtype" if invalid.endswith("dtype") else None):
        PagedKVCacheWriteFwdOp()(*inputs)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_paged_kv_cache_write_compiles_cold_to_its_operator():
    """A cold fullgraph compile owns the graph and writes the same rows as eager."""
    workload = PagedKVCacheWriteTest(5, 2, 64, 7, 3, torch.bfloat16, slots=[20, -1, 0, 6, 13])
    assert_op_owns_graph_nodes(PagedKVCacheWriteFwdOp(), *workload.gen_inputs())
    compiled_inputs = workload.gen_inputs()
    torch.compile(PagedKVCacheWriteFwdOp(), fullgraph=True)(*compiled_inputs)
    eager_inputs = workload.gen_inputs()
    PagedKVCacheWriteFwdOp()(*eager_inputs)
    torch.testing.assert_close(compiled_inputs[2], eager_inputs[2], atol=0, rtol=0)
    torch.testing.assert_close(compiled_inputs[3], eager_inputs[3], atol=0, rtol=0)


register_compile_contract(PagedKVCacheWriteFwdOp)
