"""Public paged gather semantics, including write-only views and live device metadata."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import TestBase
from tileops.attention import PagedKVCacheGatherFwdOp
from workloads.attention.paged_cache_gather import PagedKVCacheGatherWorkload


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
            [1, 33],
            (320,),
            7,
            torch.float8_e4m3fn,
            torch.float16,
            None,
            False,
            marks=pytest.mark.full,
            id="fp8-fp16-scaled",
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
    op = PagedKVCacheGatherFwdOp(out_dtype=out_dtype)
    if tune:
        op.request_tune()
    TestBase.check(workload, op, *inputs)


@pytest.mark.smoke
def test_paged_kv_cache_gather_strided_destination():
    workload = PagedKVCacheGatherWorkload([3, 17], (3, 5), 7, torch.bfloat16, starts=[1, 2])
    inputs = list(workload.gen_inputs())
    storage = torch.full((20, 3, 10), -19, dtype=inputs[5].dtype, device=inputs[5].device)
    inputs[5] = storage[..., ::2]
    inputs[5].fill_(float("nan"))
    inputs[1] = inputs[1].repeat_interleave(2, dim=1)[:, ::2]
    inputs[2] = inputs[2].repeat_interleave(2)[::2]
    inputs[3] = inputs[3].repeat_interleave(2)[::2]
    op = PagedKVCacheGatherFwdOp()
    TestBase.check(workload, op, *inputs)
    assert bool((storage[..., 1::2] == -19).all())


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.cuda_graph(op="PagedKVCacheGatherFwdOp")
def test_paged_kv_cache_gather_graph_metadata():
    workload = PagedKVCacheGatherWorkload(
        [17, 0, 33], (576,), 16, torch.float8_e4m3fn, torch.bfloat16, [1, 0, 3], table_width=5
    )
    inputs = workload.gen_inputs()
    op = PagedKVCacheGatherFwdOp(out_dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs[:5], out=inputs[5])
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*inputs[:5], out=inputs[5])
    inputs[1].copy_(inputs[1].flip(-1))
    inputs[2].copy_(torch.tensor([0, 1, 49, 50], device=inputs[2].device, dtype=torch.int32))
    inputs[3].fill_(7)
    inputs[4].fill_(0.75)
    inputs[5].fill_(float("nan"))
    graph.replay()
    TestBase.check(workload, op, *inputs, runs=lambda *args, **kwargs: inputs[5])


register_compile_contract(PagedKVCacheGatherFwdOp)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_paged_kv_cache_gather_owns_graph():
    """Tracing preserves the gather's declared operator boundary."""
    workload = PagedKVCacheGatherWorkload([17, 3], (3, 5), 7, torch.float16)
    inputs = workload.gen_inputs()
    op = PagedKVCacheGatherFwdOp()
    assert_op_owns_graph_nodes(op, *inputs[:5], out=inputs[5])


@pytest.mark.parametrize(
    "starts",
    [
        pytest.param(None, marks=pytest.mark.smoke, id="no-starts"),
        pytest.param([1, 2], marks=pytest.mark.full, id="with-starts"),
    ],
)
@pytest.mark.usefixtures("isolated_dynamo")
def test_paged_kv_cache_gather_compiles_cold(starts):
    """A cold fullgraph call serves each optional-starts specialization."""
    workload = PagedKVCacheGatherWorkload([17, 3], (3, 5), 7, torch.float16, starts=starts)
    inputs = workload.gen_inputs()
    op = PagedKVCacheGatherFwdOp()
    compiled = torch.compile(op, fullgraph=True)
    TestBase.check(workload, op, *inputs, runs=compiled)


@pytest.mark.smoke
def test_paged_kv_cache_gather_empty():
    """Zero packed rows return without launching a gather or inferring an unknown width."""
    empty = PagedKVCacheGatherWorkload([0, 0], (3, 5), 7, torch.float16)
    args = empty.gen_inputs()
    assert PagedKVCacheGatherFwdOp()(*args[:5], out=args[5]) is args[5]
    assert args[5].shape == (0, 3, 5)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("dtype", marks=pytest.mark.smoke),
        pytest.param("scale", marks=pytest.mark.full),
        pytest.param("out-dtype", marks=pytest.mark.full),
    ],
)
def test_paged_kv_cache_gather_invalid(invalid):
    workload = PagedKVCacheGatherWorkload([1], (576,), 64, torch.float8_e4m3fn, torch.bfloat16)
    inputs = list(workload.gen_inputs())
    if invalid == "dtype":
        inputs[0] = inputs[0].float()
    elif invalid == "scale":
        inputs[4] = None
    else:
        inputs[5] = inputs[5].half()
    with pytest.raises(ValueError):
        PagedKVCacheGatherFwdOp(out_dtype=torch.bfloat16)(*inputs[:5], out=inputs[5])
