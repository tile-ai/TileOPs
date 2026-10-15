"""Public paged KV cache write semantics, both cache dtypes and the skip path."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import TestBase
from tileops.ops import PagedKVCacheWriteFwdOp
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
            32,
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
            4,
            128,
            16,
            4,
            torch.float16,
            torch.float8_e4m3fn,
            [63, 0],
            False,
            marks=pytest.mark.smoke,
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
            6,
            2,
            64,
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
    k, v, k_pages, v_pages, slots, k_scale, v_scale = workload.gen_inputs()
    k_pages.fill_(3.0)
    v_pages.fill_(-2.0)
    before = (k_pages.clone(), v_pages.clone())
    op = PagedKVCacheWriteFwdOp()
    assert op(k, v, k_pages, v_pages, slots, k_scale, v_scale) is None
    assert torch.equal(k_pages, before[0]) and torch.equal(v_pages, before[1])
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
def test_paged_kv_cache_write_of_no_tokens_writes_nothing():
    """A call with no token writes neither pool; the caches it is handed are non-empty."""
    workload = PagedKVCacheWriteTest(0, 2, 64, 16, 2, torch.bfloat16)
    k, v, k_pages, v_pages, slots, k_scale, v_scale = workload.gen_inputs()
    before = (k_pages.clone(), v_pages.clone())
    op = PagedKVCacheWriteFwdOp()
    assert op(k, v, k_pages, v_pages, slots, k_scale, v_scale) is None
    assert torch.equal(k_pages, before[0]) and torch.equal(v_pages, before[1])


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
