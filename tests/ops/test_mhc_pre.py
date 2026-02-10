"""Test NativeSparseAttention operation."""

import math

import pytest
import torch

from top.ops import ManifoldConstrainedHyperConnectionPreOp


@pytest.mark.parametrize(
    ("batch, n_expand, c_x, dtype, tune"),
    [
        (1, 4, 1280, torch.bfloat16, False),
        (2, 4, 1920, torch.bfloat16, False),
        (4, 4, 2560, torch.bfloat16, False),
    ],
)
def test_mhc_pre_op(
    batch: int,
    n_expand: int,
    c_x: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:

    phi = torch.randn([n_expand * c_x, n_expand * n_expand + 2 * n_expand],
                      device="cuda",
                      dtype=torch.float32)
    x = torch.randn([batch, n_expand * c_x], device="cuda", dtype=dtype)
    b = torch.randn([n_expand * n_expand + 2 * n_expand], device="cuda", dtype=torch.float32)
    alpha_pre = torch.randn(())
    alpha_post = torch.randn(())
    alpha_res = torch.randn(())
    sinkhorn_repeat = 20
    #test_mhc_kernel = mhc_pre_kernel(batch, n_expand, c_x, dtype=torch.bfloat16)
    test_mhc_pre_op = ManifoldConstrainedHyperConnectionPreOp(
        batch, n_expand, c_x, dtype=torch.bfloat16)
    x_res, x_layer = test_mhc_pre_op.forward(phi, x, b, alpha_pre, alpha_post, alpha_res,
                                             sinkhorn_repeat)

    # check the correctness with torch...
    xsqr = x * x  # the square of x
    r_ref = torch.sqrt(xsqr.sum(dim=1)) / math.sqrt(n_expand * c_x)
    H = torch.zeros([batch, n_expand * n_expand + 2 * n_expand], device="cuda", dtype=torch.float)
    for i in range(batch):
        H[i, :] = x[i, :].float() @ phi

    H_pre_ref = H[:, :n_expand]
    H_res_ref = H[:, 2 * n_expand:]
    H_res_ref = H_res_ref.reshape(batch, n_expand, n_expand)

    b_pre_ref = b[:n_expand]
    b_res_ref = b[2 * n_expand:]
    b_res_ref = b_res_ref.reshape([n_expand, n_expand])

    H_pre_ref = torch.sigmoid(alpha_pre * H_pre_ref / r_ref.unsqueeze(-1) + b_pre_ref)
    H_res_ref = alpha_res * H_res_ref / r_ref.unsqueeze(-1).unsqueeze(-1) + b_res_ref

    eps = 0.0001
    H_res_ref_tmp = H_res_ref.max(dim=-1, keepdim=True).values

    H_res_ref = torch.exp(H_res_ref - H_res_ref_tmp)
    for _i in range(sinkhorn_repeat):
        H_res_ref = H_res_ref / (H_res_ref.sum(dim=-1, keepdim=True) + eps)
        H_res_ref = H_res_ref / (H_res_ref.sum(dim=-2, keepdim=True) + eps)
    x_in_reshaped = x.reshape([batch, n_expand, c_x])
    x_res_ref = torch.zeros([batch, n_expand, c_x], device="cuda", dtype=torch.bfloat16)
    x_layer_ref = torch.zeros([batch, c_x], device="cuda", dtype=torch.bfloat16)

    h_res_ref = H_res_ref
    h_pre_ref = H_pre_ref
    for i in range(batch):
        h_res_tmp = h_res_ref[i, :, :].float()
        h_pre_tmp = h_pre_ref[i, :].float()
        x_in_reshaped_tmp = x_in_reshaped[i, :, :].float()
        x_res_ref[i, :, :] = h_res_tmp @ x_in_reshaped_tmp
        x_layer_ref[i, :] = h_pre_tmp @ x_in_reshaped_tmp

    x_res_ref = x_res_ref.reshape(batch, n_expand * c_x)

    x_res_ref = x_res_ref.bfloat16()
    x_layer_ref = x_layer_ref.bfloat16()

    cos_sim_x_res = torch.nn.functional.cosine_similarity(x_res_ref, x_res, dim=-1, eps=1e-8)

    assert cos_sim_x_res.min() > 0.99

    cos_sim_x_layer = torch.nn.functional.cosine_similarity(x_layer_ref, x_layer, dim=-1, eps=1e-8)
    assert cos_sim_x_layer.min() > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
