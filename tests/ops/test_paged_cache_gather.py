"""Public paged gather semantics, including write-only views and live device metadata."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import TestBase
from tileops.ops import PagedKVCacheGatherFwdOp
from workloads.attention.paged_cache_gather import (
    PagedKVCacheGatherWorkload,
    paged_cache_gather_result,
)


@pytest.mark.parametrize(
    "lengths,shape,page,dtype,out_dtype,starts,tune",
    [
        pytest.param(
            [0, 17, 65],
            (3, 5),
            7,
            torch.float16,
            None,
            None,
            False,
            marks=pytest.mark.smoke,
            id="fp16-odd-page-and-row",
        ),
        pytest.param(
            [33, 0, 67],
            (8, 128),
            16,
            torch.bfloat16,
            None,
            [15, 0, 17],
            False,
            marks=pytest.mark.smoke,
            id="bf16-unaligned-starts",
        ),
        pytest.param(
            [17, 129],
            (576,),
            64,
            torch.float8_e4m3fn,
            torch.bfloat16,
            [63, 1],
            False,
            marks=pytest.mark.smoke,
            id="fp8-bf16-scaled-starts",
        ),
        pytest.param(
            [1, 33],
            (320,),
            7,
            torch.float8_e4m3fn,
            torch.float16,
            None,
            False,
            marks=pytest.mark.smoke,
            id="fp8-fp16-scaled",
        ),
        pytest.param(
            [7, 1],
            (),
            4,
            torch.float16,
            None,
            [1, 0],
            False,
            marks=pytest.mark.smoke,
            id="scalar-rows",
        ),
        pytest.param(
            [2048, 4096, 65536],
            (576,),
            64,
            torch.bfloat16,
            None,
            None,
            False,
            marks=pytest.mark.full,
            id="mixed-long-requests",
        ),
        pytest.param(
            [31, 65],
            (320,),
            16,
            torch.float16,
            None,
            [1, 3],
            True,
            marks=pytest.mark.full,
            id="tuned",
        ),
    ],
)
def test_paged_kv_cache_gather(lengths, shape, page, dtype, out_dtype, starts, tune):
    workload = PagedKVCacheGatherWorkload(lengths, shape, page, dtype, out_dtype, starts)
    inputs = workload.gen_inputs()
    op = PagedKVCacheGatherFwdOp(out_dtype=out_dtype, tune=tune)
    TestBase.check(workload, op, *inputs, runs=lambda *args: paged_cache_gather_result(op, *args))


@pytest.mark.smoke
def test_paged_kv_cache_gather_strided_destination():
    workload = PagedKVCacheGatherWorkload([3, 17], (3, 5), 7, torch.bfloat16, starts=[1, 2])
    inputs = list(workload.gen_inputs())
    storage = torch.full((20, 3, 10), -19, dtype=inputs[0].dtype, device=inputs[0].device)
    inputs[0] = storage[..., ::2]
    inputs[0].fill_(float("nan"))
    inputs[2] = inputs[2].repeat_interleave(2, dim=1)[:, ::2]
    inputs[3] = inputs[3].repeat_interleave(2)[::2]
    inputs[4] = inputs[4].repeat_interleave(2)[::2]
    op = PagedKVCacheGatherFwdOp()
    TestBase.check(workload, op, *inputs, runs=lambda *args: paged_cache_gather_result(op, *args))
    assert bool((storage[..., 1::2] == -19).all())


@pytest.mark.smoke
@pytest.mark.cuda_only
def test_paged_kv_cache_gather_graph_metadata():
    workload = PagedKVCacheGatherWorkload(
        [17, 0, 33], (576,), 16, torch.float8_e4m3fn, torch.bfloat16, [1, 0, 3], table_width=5
    )
    inputs = workload.gen_inputs()
    op = PagedKVCacheGatherFwdOp(out_dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*inputs)
    inputs[2].copy_(inputs[2].flip(-1))
    inputs[3].copy_(torch.tensor([0, 1, 49, 50], device=inputs[3].device, dtype=torch.int32))
    inputs[4].fill_(7)
    inputs[5].fill_(0.75)
    inputs[0].fill_(float("nan"))
    graph.replay()
    TestBase.check(workload, op, *inputs, runs=lambda *args: inputs[0])


register_compile_contract(PagedKVCacheGatherFwdOp)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_paged_kv_cache_gather_compile_and_empty():
    workload = PagedKVCacheGatherWorkload([17, 3], (3, 5), 7, torch.float16)
    inputs = workload.gen_inputs()
    op = PagedKVCacheGatherFwdOp()
    assert_op_owns_graph_nodes(op, *inputs)
    compiled = torch.compile(op, fullgraph=True)
    TestBase.check(
        workload, op, *inputs, runs=lambda *args: paged_cache_gather_result(compiled, *args)
    )
    starts = torch.tensor([1, 2], dtype=torch.int32, device=inputs[0].device)
    with_starts = (*inputs[:4], starts, inputs[5])
    TestBase.check(
        workload, op, *with_starts, runs=lambda *args: paged_cache_gather_result(compiled, *args)
    )
    empty = PagedKVCacheGatherWorkload([0, 0], (3, 5), 7, torch.float16)
    args = empty.gen_inputs()
    assert compiled(*args) is None
    assert args[0].shape == (0, 3, 5)


@pytest.mark.smoke
@pytest.mark.parametrize("invalid", ["scale", "out-dtype"])
def test_paged_kv_cache_gather_invalid(invalid):
    workload = PagedKVCacheGatherWorkload([1], (576,), 64, torch.float8_e4m3fn, torch.bfloat16)
    inputs = list(workload.gen_inputs())
    if invalid == "scale":
        inputs[-1] = None
    else:
        inputs[0] = inputs[0].half()
    with pytest.raises(ValueError):
        PagedKVCacheGatherFwdOp(out_dtype=torch.bfloat16)(*inputs)
