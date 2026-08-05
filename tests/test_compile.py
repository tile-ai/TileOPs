# This test validates the compatibility of TileOps operators with torch.compile().
# Check: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html

import pytest
import torch

from tests.compile_contract import register_compile_contract
from tests.ops.attention.test_mha import MhaFwdTest
from tests.test_base import FixtureBase
from tileops.ops import MultiHeadAttentionFwdOp

register_compile_contract(MultiHeadAttentionFwdOp)


class MhaCompileFixture(FixtureBase):
    PARAMS = [
        ("B, S, H, D, causal, dtype", [
            (8, 1024, 32, 128, False, torch.float16),
            (4, 512, 16, 64, True, torch.bfloat16),
        ]),
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
def test_mha_compiled_returns_output_lse_pair():
    """A cold fullgraph trace returns the same ``(output, lse)`` pair as eager."""
    B, S, H, D = 1, 128, 8, 64
    op = MultiHeadAttentionFwdOp(B, H, S, D, False)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    output, lse = torch.compile(op, fullgraph=True)(q, k, v)

    assert output.shape == q.shape
    assert lse is None
    torch.testing.assert_close(output, op(q, k, v)[0])


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
