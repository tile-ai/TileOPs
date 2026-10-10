"""Request-local pipeline ownership and replay tests for Hopper paged attention."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.kernels.attention.call_spec import AttentionCall
from tileops.kernels.attention.gqa.paged_mixed import GQAPagedMixedKernel
from tileops.ops import GQAPagedFwdOp
from workloads.attention.gqa.paged import GQAPagedFwdWorkload


@pytest.mark.smoke
@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, "gqa_paged_mixed_kernel"),
        ({"batch": 1, "is_uniform": True}, "gqa_paged_mixed_kernel"),
        ({"arch": 80}, "gqa_paged_varlen_kernel"),
        ({"arch": 89}, "gqa_paged_varlen_kernel"),
        ({"max_seqlen_q": 128}, "gqa_paged_mixed_kernel"),
        ({"batch": 256, "max_seqlen_q": 256}, "gqa_paged_mixed_kernel"),
        ({"max_seqlen_q": 128, "window_size_left": 128}, "gqa_paged_mixed_kernel"),
        ({"max_seqlen_q": 128, "dim": 64}, "gqa_paged_varlen_kernel"),
        ({"max_seqlen_q": 128, "heads_kv": 1}, "gqa_paged_varlen_kernel"),
        ({"max_pages_per_req": 8}, "gqa_paged_varlen_kernel"),
        ({"fuse_rope": True}, "gqa_paged_varlen_kernel"),
        ({"window_size_left": 128}, "gqa_paged_varlen_kernel"),
        ({"sm_scale": -0.125}, "gqa_paged_varlen_kernel"),
        ({"dim": 256}, "gqa_paged_varlen_kernel"),
        ({"page_size": 37}, "gqa_paged_varlen_kernel"),
    ],
)
def test_paged_mixed_dispatch_regions(changes, expected):
    facts = dict(
        arch=90,
        batch=8,
        heads=32,
        heads_kv=8,
        dim=128,
        dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        is_uniform=False,
        max_seqlen_q=1024,
        seqlen_kv=32768,
        page_size=64,
        max_pages_per_req=64,
        is_causal=True,
        sm_scale=128**-0.5,
    )
    facts.update(changes)
    assert GQAPagedFwdOp().select_implementation("gqa_paged", AttentionCall(**facts)) == expected


def _workload(page_size, dtype, dim=128):
    width = (2049 + page_size - 1) // page_size
    return GQAPagedFwdWorkload(
        16,
        4,
        dim,
        [129, 1, 31, 32, 33, 0],
        [2049, 2049, 35, 0, 17, 0],
        page_size,
        width,
        6 * width,
        dtype,
    )


def _poison(workload, inputs):
    for request, length in enumerate(workload.cache_lens):
        slots = torch.arange(
            length, workload.max_pages_per_req * workload.page_size, device=inputs[0].device
        )
        pages = inputs[3][request, slots // workload.page_size]
        inputs[1][pages, slots % workload.page_size] = float("nan")
        inputs[2][pages, slots % workload.page_size] = float("inf")


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize("page_size", [48, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_mixed_request_ownership(page_size, dtype):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper mixed pipeline")
    workload = _workload(page_size, dtype)
    inputs = workload.gen_inputs()
    _poison(workload, inputs)
    op = GQAPagedFwdOp()
    TestBase.check(workload, op, *inputs)
    assert any(isinstance(kernel, GQAPagedMixedKernel) for kernel in op._dispatched.values())


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "page_size,is_causal,heads_kv,softcap", [(48, True, 2, 0.0), (128, False, 4, 3.0)]
)
def test_paged_mixed_narrow_heads_and_softcap(page_size, is_causal, heads_kv, softcap):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper mixed pipeline")
    width = (2049 + page_size - 1) // page_size
    cut = 128 // (16 // heads_kv)
    workload = GQAPagedFwdWorkload(
        16,
        heads_kv,
        64,
        [129, 1, cut - 1, cut, cut + 1, 0],
        [2049, 13, 5, 0, 17, 0],
        page_size,
        width,
        6 * width,
        torch.bfloat16,
        is_causal=is_causal,
        softcap=softcap,
    )
    inputs = workload.gen_inputs()
    _poison(workload, inputs)
    op = GQAPagedFwdOp(is_causal=is_causal, softcap=softcap)
    TestBase.check(workload, op, *inputs)
    assert any(isinstance(kernel, GQAPagedMixedKernel) for kernel in op._dispatched.values())


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize("page_size", [48, 128])
def test_paged_mixed_graph_changes_pipeline_ownership(page_size):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper mixed pipeline")
    workload = _workload(page_size, torch.bfloat16)
    inputs = workload.gen_inputs()
    original_k, original_v = inputs[1].clone(), inputs[2].clone()
    op = GQAPagedFwdOp()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    # Keep shapes/addresses fixed while requests cross the cut in both directions.
    for q_lens, cache_lens in [
        ([0, 33, 32, 31, 1, 129], [0, 17, 2049, 35, 0, 2049]),
        ([129, 1, 31, 32, 33, 0], [0, 2049, 35, 0, 17, 0]),
        ([226, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    ]:
        workload.q_lens, workload.cache_lens = q_lens, cache_lens
        inputs[1].copy_(original_k)
        inputs[2].copy_(original_v)
        inputs[4].copy_(torch.tensor(cache_lens, dtype=torch.int32, device=inputs[0].device))
        lengths = torch.tensor([0, *q_lens], dtype=torch.int32, device=inputs[0].device)
        inputs[5].copy_(lengths.cumsum(0))
        _poison(workload, inputs)
        graph.replay()
        TestBase.check(workload, op, *inputs, runs=lambda *args: output)


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "page_size,group,total_q,dtype,options",
    [
        # Partial output warps, including a non-power-of-two head group.
        (64, 1, 8, torch.float16, {}),
        (64, 3, 8, torch.float16, {}),
        (16, 4, 32, torch.float16, {}),
        (48, 8, 8, torch.bfloat16, {}),
        (64, 16, 4, torch.float16, {"softcap": 50.0}),
        (256, 4, 32, torch.bfloat16, {"softcap": 3.0, "sm_scale": 0.25}),
        # The fast tanh/scale path must not underflow or lose large-cap logits.
        (64, 4, 8, torch.float16, {"softcap": 1e-38}),
        (64, 4, 8, torch.bfloat16, {"softcap": 1e38}),
        (64, 4, 8, torch.float16, {"window_size_left": 19}),
        (
            128,
            8,
            8,
            torch.bfloat16,
            {"is_causal": False, "window_size_left": 19, "window_size_right": 5},
        ),
        # Window tiling on both sides of the index-legalization boundary,
        # with split and unsplit execution, including a zero-width window.
        (64, 4, 32, torch.bfloat16, {"window_size_left": 0}),
        (64, 4, 32, torch.bfloat16, {"window_size_left": 63}),
        (64, 4, 32, torch.bfloat16, {"window_size_left": 64}),
        (64, 4, 8, torch.bfloat16, {"window_size_left": 255}),
        (64, 4, 8, torch.bfloat16, {"window_size_left": 256}),
        (64, 4, 32, torch.bfloat16, {"is_causal": False, "window_size_right": 0}),
    ],
)
def test_paged_small_ragged_mma(page_size, group, total_q, dtype, options):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper paged MMA pipeline")
    width = (2049 + page_size - 1) // page_size
    # Includes an empty request, a fully masked query, a partial final page,
    # and multi-token requests despite total_q being in the decode region.
    workload = GQAPagedFwdWorkload(
        group * 8,
        8,
        128,
        [0, 1, 2, total_q - 3],
        [0, 2049, 0, 17],
        page_size,
        width,
        4 * width,
        dtype,
        **options,
    )
    inputs = workload.gen_inputs()
    _poison(workload, inputs)
    op = GQAPagedFwdOp(**options)
    TestBase.check(workload, op, *inputs)
    assert any(isinstance(kernel, GQAPagedMixedKernel) for kernel in op._dispatched.values())


@pytest.mark.smoke
@pytest.mark.cuda_only
def test_paged_small_empty_queries():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper paged MMA pipeline")
    workload = _workload(64, torch.bfloat16)
    workload.q_lens = [0] * workload.batch
    inputs = workload.gen_inputs()
    TestBase.check(workload, GQAPagedFwdOp(), *inputs)


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize("total_q", [8, 32])
def test_paged_small_graph_changes_request_lengths(total_q):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper paged MMA pipeline")
    workload = GQAPagedFwdWorkload(
        32,
        8,
        128,
        [1] * total_q,
        [2049] * total_q,
        64,
        33,
        total_q * 33,
        torch.bfloat16,
    )
    inputs = workload.gen_inputs()
    original_k, original_v = inputs[1].clone(), inputs[2].clone()
    op = GQAPagedFwdOp()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    for q_lens, cache_lens in [
        ([0] * (total_q - 1) + [total_q], [0] * (total_q - 1) + [17]),
        ([total_q] + [0] * (total_q - 1), [0] * total_q),
        ([1] * total_q, [2049] * total_q),
    ]:
        workload.q_lens, workload.cache_lens = q_lens, cache_lens
        inputs[1].copy_(original_k)
        inputs[2].copy_(original_v)
        inputs[4].copy_(torch.tensor(cache_lens, dtype=torch.int32, device=inputs[0].device))
        lengths = torch.tensor([0, *q_lens], dtype=torch.int32, device=inputs[0].device)
        inputs[5].copy_(lengths.cumsum(0))
        _poison(workload, inputs)
        graph.replay()
        TestBase.check(workload, op, *inputs, runs=lambda *args: output)


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize("page_size", [48, 64])
def test_paged_persistent_queue_reuses_slots(page_size):
    """Several waves mix long, single-token, packed, and empty requests."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper paged pipeline")
    width = (4099 + page_size - 1) // page_size
    workload = GQAPagedFwdWorkload(
        32,
        8,
        128,
        [1000, 1, 16, 33, 0],
        [4099, 4099, 35, 0, 0],
        page_size,
        width,
        5 * width,
        torch.bfloat16,
    )
    inputs = workload.gen_inputs()
    _poison(workload, inputs)
    op = GQAPagedFwdOp()
    # Counter reset and metadata-slot reuse must remain correct across launches.
    for _ in range(3):
        TestBase.check(workload, op, *inputs)


@pytest.mark.smoke
@pytest.mark.cuda_only
def test_paged_independent_graph_workspaces():
    """Two graphs captured from one Op may replay concurrently."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper paged pipeline")
    workload = GQAPagedFwdWorkload(
        32,
        8,
        128,
        [0, 1, 2, 5],
        [0, 2049, 17, 0],
        64,
        33,
        4 * 33,
        torch.bfloat16,
    )
    inputs = [workload.gen_inputs(), workload.gen_inputs()]
    op = GQAPagedFwdOp()
    for args in inputs:
        _poison(workload, args)
        op(*args)
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outputs = [], []
    for stream, args in zip(streams, inputs, strict=True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(op(*args))
        graphs.append(graph)
    for _ in range(8):
        for stream, graph in zip(streams, graphs, strict=True):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    for args, output in zip(inputs, outputs, strict=True):
        TestBase.check(workload, op, *args, runs=lambda *unused, result=output: result)


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize("query,window", [(4, -1), (5, -1), (5, 19)])
def test_paged_small_pipeline_boundary(query, window):
    """Cross the whole-call boundary without dropping window masks or tail guards."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper paged pipeline")
    workload = GQAPagedFwdWorkload(
        32,
        8,
        128,
        [query] * 8,
        [2049, 17, 0, 33, 7, 129, 65, 1],
        64,
        33,
        8 * 33,
        torch.bfloat16,
        window_size_left=window,
    )
    inputs = workload.gen_inputs()
    _poison(workload, inputs)
    TestBase.check(workload, GQAPagedFwdOp(window_size_left=window), *inputs)


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "group,page_size,dtype", [(8, 64, torch.bfloat16), (16, 48, torch.float16)]
)
def test_paged_large_packed_graph_boundary(group, page_size, dtype):
    """Multi-tile packed requests cross into prefill on replay, including masked rows."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper paged pipeline")
    width = (2049 + page_size - 1) // page_size
    workload = GQAPagedFwdWorkload(
        group * 8,
        8,
        128,
        [65, 64, 63, 1, 0, 0, 0, 0],
        [67, 129, 0, 2049, 0, 0, 0, 0],
        page_size,
        width,
        8 * width,
        dtype,
    )
    inputs = workload.gen_inputs()
    _poison(workload, inputs)
    op = GQAPagedFwdOp()
    TestBase.check(workload, op, *inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    for lengths in ([64, 65, 1, 63, 0, 0, 0, 0], [65, 64, 63, 1, 0, 0, 0, 0]):
        workload.q_lens = list(lengths)
        cu_q = torch.tensor([0, *lengths], device=inputs[0].device, dtype=torch.int32).cumsum(0)
        inputs[5].copy_(cu_q)
        graph.replay()
        TestBase.check(workload, op, *inputs, runs=lambda *unused: output)
