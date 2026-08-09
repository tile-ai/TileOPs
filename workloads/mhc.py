import math

import torch

from workloads.workload_base import WorkloadBase


class MHCPreWorkload(WorkloadBase):

    def __init__(self, batch: int, n_expand: int, c_x: int, dtype: torch.dtype):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        phi = torch.randn([n_expand * c_x, n_expand * n_expand + 2 * n_expand],
                          device="cuda",
                          dtype=torch.float32)
        x = torch.randn([batch, n_expand * c_x], device="cuda", dtype=torch.bfloat16)
        b = torch.randn([n_expand * n_expand + 2 * n_expand], device="cuda", dtype=torch.float32)
        alpha_pre = torch.randn(())
        alpha_post = torch.randn(())
        alpha_res = torch.randn(())
        sinkhorn_repeat = 20
        eps = 0.02
        return phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, eps

    def ref_program(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor,
                    alpha_pre, alpha_post, alpha_res,
                    sinkhorn_repeat: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        return mhc_pre_ref(
            self.batch, self.n_expand, self.c_x,
            phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, eps,
        )


class MHCPostWorkload(WorkloadBase):

    def __init__(self, batch: int, n_expand: int, c_x: int, dtype: torch.dtype):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_layer_out = torch.randn([batch, c_x], device="cuda", dtype=self.dtype)
        h_post = torch.randn([batch, n_expand], device="cuda", dtype=torch.float32)
        x_res = torch.randn([batch, n_expand * c_x], device="cuda", dtype=self.dtype)
        return x_layer_out, h_post, x_res

    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
            self.batch, self.n_expand * self.c_x) + x_res.float()
        return x_out_ref.bfloat16()


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
