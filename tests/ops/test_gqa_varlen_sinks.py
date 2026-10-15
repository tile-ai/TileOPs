"""Per-head sink normalization across packed attention implementations."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes
from tests.workload_test_base import TestBase
from tileops.attention import GQAVarlenFwdOp
from workloads.attention.gqa.varlen import GQAVarlenSinkWorkload


@pytest.mark.parametrize(
    "dtype,dim,q_lens,kv_lens,options",
    [
        pytest.param(
            torch.float16,
            128,
            [129, 1, 0, 17],
            [257, 33, 0, 17],
            {},
            marks=pytest.mark.smoke,
            id="fp16-persistent",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            [129, 1, 0, 17],
            [257, 33, 0, 17],
            {"window_size_left": 64},
            marks=pytest.mark.smoke,
            id="bf16-window",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            [129, 1],
            [257, 33],
            {},
            marks=pytest.mark.smoke,
            id="fp8-persistent",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            [129, 1],
            [257, 33],
            {"rotary_dim": 64},
            marks=pytest.mark.full,
            id="partial-rope",
        ),
        pytest.param(
            torch.float16,
            128,
            [129, 1],
            [257, 33],
            {"rotary_dim": 128, "rope_layout": "interleaved", "window_size_left": 64},
            marks=pytest.mark.full,
            id="rope-window",
        ),
        pytest.param(
            torch.bfloat16, 128, [17, 1], [0, 0], {}, marks=pytest.mark.full, id="empty-kv"
        ),
        pytest.param(
            torch.float16,
            128,
            [17, 1],
            [33, 17],
            {"sm_scale": -0.125},
            marks=pytest.mark.full,
            id="negative-scale",
        ),
        pytest.param(
            torch.float16,
            128,
            [17, 1],
            [33, 17],
            {"sm_scale": 0.0, "softcap": 5.0},
            marks=pytest.mark.full,
            id="zero-scale-softcap",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            [17, 129],
            [33, 257],
            {"is_causal": False, "window_size_left": 0, "window_size_right": 32},
            marks=pytest.mark.full,
            id="right-window",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            [17, 1],
            [33, 17],
            {"rotary_dim": 64, "out_dtype": torch.float16},
            marks=pytest.mark.full,
            id="fp8-rope",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            [129, 1],
            [257, 33],
            {"sm_scale": 0.125, "softcap": 5.0},
            marks=pytest.mark.full,
            id="bf16-softcap",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            128,
            [129, 1],
            [257, 33],
            {"sm_scale": 0.125, "softcap": 5.0},
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="fp8-softcap",
        ),
        pytest.param(
            torch.float16,
            256,
            [300, 5],
            [100, 3],
            {"window_size_left": 64},
            marks=pytest.mark.full,
            id="fp16-empty-rows-general",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            [300, 5],
            [100, 3],
            {"window_size_left": 64},
            marks=pytest.mark.full,
            id="bf16-empty-rows-persistent",
        ),
    ],
)
def test_varlen_sinks(dtype, dim, q_lens, kv_lens, options):
    """Normalize per-head sinks across the supported attention paths."""
    w, op = _sink_case(dtype, dim, q_lens, kv_lens, options)
    TestBase.check(w, op, *w.gen_inputs())


@pytest.mark.sm90
@pytest.mark.parametrize(
    "heads,has_sinks",
    [
        pytest.param(8, False, marks=pytest.mark.smoke, id="no-sink"),
        pytest.param(8, True, marks=pytest.mark.full, id="sinks-persistent"),
        pytest.param(6, True, marks=pytest.mark.full, id="sinks-general"),
    ],
)
def test_varlen_fp8_empty_rows(heads, has_sinks):
    """SM90 FP8 rows with no visible key return zero instead of dividing zero by zero."""
    w, op = _sink_case(
        torch.float8_e4m3fn, 128, [300, 5], [100, 3], {"window_size_left": 64, "heads": heads}
    )
    inputs = list(w.gen_inputs())
    if not has_sinks:
        inputs[-1] = None
    TestBase.check(w, op, *inputs)


def _sink_case(dtype, dim, q_lens, kv_lens, options):
    options = dict(options)
    heads = options.pop("heads", 8)
    out_dtype = options.get("out_dtype", torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype)
    w = GQAVarlenSinkWorkload(
        len(q_lens),
        q_lens,
        kv_lens,
        heads,
        2,
        dim,
        options.get("is_causal", True),
        options.get("window_size_left", -1),
        options.get("window_size_right", -1),
        dtype,
        sm_scale=options.get("sm_scale"),
        softcap=options.get("softcap"),
        out_dtype=out_dtype,
        rotary_dim=options.get("rotary_dim"),
        rope_layout=options.get("rope_layout", "neox"),
        has_sinks=True,
    )
    op = GQAVarlenFwdOp(
        **options,
        **({} if "out_dtype" in options else {"out_dtype": out_dtype}),
        pos_encoding_mode="rope" if "rotary_dim" in options else "none",
    )
    return w, op


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_varlen_sinks_compile():
    """Compilation retains the Op boundary and per-head sink normalization."""
    w, op = _sink_case(torch.float16, 128, [129, 1, 0, 17], [257, 33, 0, 17], {})
    inputs = w.gen_inputs()
    assert_op_owns_graph_nodes(op, *inputs)
    TestBase.check(w, op, *inputs, runs=torch.compile(op, fullgraph=True))


@pytest.mark.smoke
def test_varlen_sinks_rejects_non_fp32_dtype():
    """The sink contract rejects a vector with a dtype other than float32."""
    w, op = _sink_case(torch.float16, 128, [17, 1], [33, 17], {})
    inputs = list(w.gen_inputs())
    inputs[-1] = inputs[-1].to(torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        op(*inputs)


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "dtype,options",
    [
        pytest.param(torch.bfloat16, {}, marks=pytest.mark.smoke, id="sink-bf16"),
        pytest.param(
            torch.float8_e4m3fn, {}, marks=[pytest.mark.smoke, pytest.mark.sm90], id="sink-fp8"
        ),
        pytest.param(torch.bfloat16, {"rotary_dim": 64}, marks=pytest.mark.full, id="partial-rope"),
        pytest.param(
            torch.bfloat16,
            {"rotary_dim": 128, "rope_layout": "interleaved", "window_size_left": 64},
            marks=pytest.mark.full,
            id="rope-window",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            {"rotary_dim": 64, "out_dtype": torch.float16},
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="fp8-rope",
        ),
    ],
)
def test_varlen_sinks_graph_replay(dtype, options):
    """A captured call reads updated sink values, including calls with RoPE."""
    w, op = _sink_case(dtype, 128, [129, 1], [257, 33], options)
    inputs = w.gen_inputs()
    TestBase.check(w, op, *inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(*inputs)
    inputs[-1][4:].add_(1)
    graph.replay()
    TestBase.check(w, op, *inputs, runs=lambda *args: output)
