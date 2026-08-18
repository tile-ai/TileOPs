# This test validates the compatibility of TileOps operators with torch.compile().
# Check: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.ops.attention.test_mha import MhaFwdTest
from tests.test_base import FixtureBase
from tileops.ops import GroupedQueryAttentionPrefillDenseFwdOp, MultiHeadAttentionFwdOp
from tileops.ops.attention.gqa import _gqa_prefill_dense_fwd_fake

register_compile_contract(MultiHeadAttentionFwdOp)
register_compile_contract(GroupedQueryAttentionPrefillDenseFwdOp)


class MhaCompileFixture(FixtureBase):
    PARAMS = [
        (
            "B, S, H, D, causal, dtype",
            [
                (8, 1024, 32, 128, False, torch.float16),
                (4, 512, 16, 64, True, torch.bfloat16),
            ],
        ),
    ]


@pytest.mark.full
@pytest.mark.usefixtures("isolated_dynamo")
@MhaCompileFixture
def test_mha_kernel_compile(B: int, S: int, H: int, D: int, causal: bool, dtype: torch.dtype):
    test = MhaFwdTest(B, H, S, D, causal, dtype)
    op = MultiHeadAttentionFwdOp(B, H, S, D, causal)
    compiled_op = torch.compile(op, fullgraph=True)
    inputs = test.gen_inputs()
    test.check(compiled_op, *inputs, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_mha_cold_fullgraph_trace_matches_eager():
    """The kernel is built inside the custom op, so a cold trace must still match."""
    B, S, H, D = 1, 128, 8, 64
    op = MultiHeadAttentionFwdOp(B, H, S, D, False)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    output = torch.compile(op, fullgraph=True)(q, k, v)

    assert output.shape == q.shape
    torch.testing.assert_close(output, op(q, k, v))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_dense_gqa_traced_graph_holds_only_the_op_node():
    """Changing the Dense target or specialization cannot change graph identity."""
    batch, seq_len, heads, heads_kv, dim = 1, 128, 8, 2, 64
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch, heads, heads_kv, seq_len, dim, is_causal=False
    )
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    assert_op_owns_graph_nodes(op, q, k, v)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_dense_gqa_non_contiguous_cold_fullgraph_matches_eager():
    """The fake promises manifest shape/dtype while eager normalization fixes strides."""
    batch, seq_len, heads, heads_kv, dim = 1, 128, 8, 2, 64
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch, heads, heads_kv, seq_len, dim, is_causal=False
    )
    q = torch.randn(
        batch, seq_len, heads, dim * 2, device="cuda", dtype=torch.float16
    )[..., ::2]
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    assert not q.is_contiguous()

    output = torch.compile(op, fullgraph=True)(q, k, v)

    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert output.is_contiguous()
    torch.testing.assert_close(output, op(q, k, v), atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_dense_gqa_present_optional_inputs_cold_fullgraph_matches_eager():
    """The all-present Optional[Tensor] signature keeps the same Op-owned graph node."""
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("native FP8 prefill requires SM90 and float8_e4m3fn")
    batch, seq_len, heads, heads_kv, dim = 1, 65, 8, 2, 128
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch,
        heads,
        heads_kv,
        seq_len,
        dim,
        is_causal=False,
        dtype=torch.float16,
        fuse_rope=True,
    )
    q = torch.randn(batch, seq_len, heads, dim, device="cuda").clamp(-2, 2).to(fp8)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda").clamp(-2, 2).to(fp8)
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda").clamp(-2, 2).to(fp8)
    scales = torch.ones(batch, heads_kv, device="cuda", dtype=torch.float32)
    rope_cos = torch.ones(seq_len, dim // 2, device="cuda", dtype=torch.float16)
    rope_sin = torch.zeros_like(rope_cos)
    inputs = (q, k, v, scales, scales, scales, rope_cos, rope_sin)

    output = torch.compile(op, fullgraph=True)(*inputs)

    assert output.shape == q.shape
    assert output.dtype == torch.float16
    torch.testing.assert_close(output, op(*inputs), atol=8e-2, rtol=2e-2)


@pytest.mark.smoke
@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable"
)
def test_dense_gqa_fake_uses_manifest_shape_and_selected_output_dtype():
    """The Op-owned fake, not an internal kernel fake, defines graph metadata."""
    batch, seq_len, heads, heads_kv, dim = 1, 7, 8, 2, 128
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch,
        heads,
        heads_kv,
        seq_len,
        dim,
        is_causal=False,
        dtype=torch.bfloat16,
    )
    q = torch.empty(batch, seq_len, heads, dim, dtype=torch.float8_e4m3fn)
    k = torch.empty(batch, seq_len, heads_kv, dim, dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)

    output = _gqa_prefill_dense_fwd_fake(
        q, k, v, None, None, None, None, None, op._instance_key
    )

    assert output.shape == q.shape
    assert output.dtype == torch.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
