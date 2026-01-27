"""Test NativeSparseAttention operation."""

import pytest
import torch

from top.ops import ManifoldConstrainedHyperConnectionPostOp


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(12345)

@pytest.mark.parametrize(
    ("batch, n_expand, c_x, dtype, tune"),
    [
        (1, 4, 1280, torch.bfloat16, False),
        (2, 4, 1920, torch.bfloat16, False),
        (4, 4, 2560, torch.bfloat16, False),
    ],
)
def test_mhc_post_op(
    batch: int,
    n_expand: int,
    c_x: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    x_layer_out = torch.randn([batch, c_x], device="cuda", dtype=dtype)
    h_post = torch.randn([batch, n_expand], device="cuda", dtype=torch.float32)
    x_res = torch.randn([batch, n_expand * c_x], device="cuda", dtype=dtype)

    x_out_ref = torch.zeros([batch, n_expand * c_x], device="cuda", dtype=dtype)
    for i in range(batch):
        x_out_ref[i, :] = (h_post[i, :].reshape(n_expand, 1).float() @ x_layer_out[i, :].reshape(1,
                                                                                                 c_x).float()).reshape(
            n_expand * c_x) + x_res[i, :].float()

    test_mhc_post_op = ManifoldConstrainedHyperConnectionPostOp(batch, n_expand, c_x, dtype="bfloat16")
    x_out = test_mhc_post_op.forward(x_layer_out, h_post, x_res)
    print(x_out_ref)
    print(x_out)

    x_out_diff = x_out_ref - x_out
    cos_sim_x_out = torch.nn.functional.cosine_similarity(x_out_ref, x_out, dim=-1, eps=1e-8)
    print(x_out_diff.abs())
    assert cos_sim_x_out.min() > 0.99

