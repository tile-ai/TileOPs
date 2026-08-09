"""Pure-PyTorch references for the Multi-Head Compression (MHC) stages.

Shared by the MHC tests and benchmarks so each stage has one oracle instead
of a verbatim copy on both sides.
"""

import math

import torch


def mhc_pre_ref(
    batch: int,
    n_expand: int,
    c_x: int,
    phi: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    alpha_pre,
    alpha_post,
    alpha_res,
    sinkhorn_repeat: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the MHC pre-projection stage.

    Returns:
        ``(x_res_ref, x_layer_ref)``, both bfloat16.
    """

    xsqr = x * x
    norm_eps = 0.0001
    r_ref = torch.sqrt(xsqr.sum(dim=1)) / math.sqrt(n_expand * c_x) + norm_eps
    H = torch.zeros([batch, n_expand * n_expand + 2 * n_expand],
                    device=x.device, dtype=torch.float)
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

    H_res_ref_tmp = H_res_ref.max(dim=-1, keepdim=True).values

    H_res_ref = torch.exp(H_res_ref - H_res_ref_tmp)
    for _i in range(sinkhorn_repeat):
        H_res_ref = H_res_ref / (H_res_ref.sum(dim=-1, keepdim=True) + eps)
        H_res_ref = H_res_ref / (H_res_ref.sum(dim=-2, keepdim=True) + eps)
    x_in_reshaped = x.reshape([batch, n_expand, c_x])
    x_res_ref = torch.zeros([batch, n_expand, c_x], device=x.device, dtype=torch.bfloat16)
    x_layer_ref = torch.zeros([batch, c_x], device=x.device, dtype=torch.bfloat16)

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
    return x_res_ref, x_layer_ref



def mhc_post_ref(
    batch: int,
    n_expand: int,
    c_x: int,
    x_layer_out: torch.Tensor,
    h_post: torch.Tensor,
    x_res: torch.Tensor,
) -> torch.Tensor:
    """Reference for the MHC post-projection stage."""

    x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
        batch, n_expand * c_x) + x_res.float()
    x_out_ref = x_out_ref.bfloat16()
    return x_out_ref
