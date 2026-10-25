"""Dense decode with 256-wide heads, including short caches and fused RoPE."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes
from tests.workload_test_base import TestBase
from tileops.attention import GQADenseFwdOp
from workloads.attention.gqa.dense import GQADensePrefillWorkload


@pytest.mark.parametrize(
    "batch, heads, kv, dtype, softcap, rotary_dim, rope_layout, tune, dim",
    [
        pytest.param(
            1, 8, 63, torch.float16, 0.0, None, "neox", False, 256, marks=pytest.mark.smoke
        ),
        pytest.param(
            16, 8, 8192, torch.bfloat16, 0.0, None, "neox", False, 256, marks=pytest.mark.smoke
        ),
        pytest.param(
            2, 71, 129, torch.float16, 0.0, None, "neox", False, 256, marks=pytest.mark.full
        ),
        pytest.param(
            1, 8, 8193, torch.bfloat16, 50.0, None, "neox", False, 256, marks=pytest.mark.full
        ),
        pytest.param(
            3, 8, 129, torch.float16, 0.0, None, "neox", True, 256, marks=pytest.mark.full
        ),
        pytest.param(
            1,
            8,
            129,
            torch.float16,
            0.0,
            64,
            "neox",
            False,
            256,
            marks=[pytest.mark.full, pytest.mark.sm90],
        ),
        pytest.param(
            2,
            8,
            513,
            torch.bfloat16,
            2.0,
            256,
            "interleaved",
            False,
            256,
            marks=[pytest.mark.full, pytest.mark.sm90],
        ),
        # The same missing softcap also affected the pre-existing D=128 RoPE path.
        pytest.param(
            2,
            8,
            513,
            torch.bfloat16,
            2.0,
            128,
            "interleaved",
            False,
            128,
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="d128-rope-softcap-regression",
        ),
    ],
)
def test_gqa_dense_decode_wide_heads(
    batch, heads, kv, dtype, softcap, rotary_dim, rope_layout, tune, dim
):
    workload = GQADensePrefillWorkload(
        batch,
        1,
        kv,
        heads,
        1,
        dim,
        dtype,
        softcap=softcap,
        rotary_dim=rotary_dim,
        rope_layout=rope_layout,
    )
    op = GQADenseFwdOp(
        is_causal=True,
        softcap=softcap,
        pos_encoding_mode="none" if rotary_dim is None else "rope",
        rotary_dim=rotary_dim,
        rope_layout=rope_layout,
    )
    if tune:
        op.request_tune()
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_gqa_dense_decode_d256_compile():
    """One compiled Op serves short, split and trailing partial KV tiles."""
    workload = GQADensePrefillWorkload(1, 1, 1, 8, 1, 256, torch.bfloat16)
    inputs = workload.gen_inputs()
    op = GQADenseFwdOp(is_causal=True)
    assert_op_owns_graph_nodes(op, *inputs)
    compiled = torch.compile(op, fullgraph=True)
    TestBase.check(workload, op, *inputs, runs=compiled)
    for kv in (129, 63):
        changed = GQADensePrefillWorkload(1, 1, kv, 8, 1, 256, torch.bfloat16)
        TestBase.check(changed, op, *changed.gen_inputs(), runs=compiled)
