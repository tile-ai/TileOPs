import math
from typing import Tuple

import torch

from benchmarks.benchmark import Benchmark
from top.ops import ManifoldConstrainedHyperConnectionPreOp


class ManifoldConstrainedHyperConnectionPreBenchmark(Benchmark):

    op_type = ManifoldConstrainedHyperConnectionPreOp

    def __init__(self, batch: int, n_expand: int, c_x: int, dtype: torch.dtype):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype

    @property
    def total_flops(self) -> float:
        flops = 2 * self.batch * (
            (self.n_expand * self.n_expand * self.c_x * self.c_x) *
            (self.n_expand * self.n_expand + 2 * self.n_expand) + self.n_expand * self.c_x)
        return flops

    @property
    def total_memory(self) -> float:
        return (self.n_expand * 3 + 1) * self.c_x + (self.n_expand * self.c_x) * (
            self.n_expand * self.n_expand + 2 * self.n_expand)

    def gen_inputs(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               int]:
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
        return phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat

    def ref_program(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor, alpha_pre: int,
                    alpha_post: int, alpha_res: int,
                    sinkhorn_repeat: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        xsqr = x * x  # the square of x
        r_ref = torch.sqrt(xsqr.sum(dim=1)) / math.sqrt(n_expand * c_x)
        H = x.float() @ phi

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

        h_res_ref = H_res_ref
        h_pre_ref = H_pre_ref

        h_res_ref.reshape(batch)
        x_res_ref = torch.bmm(h_res_ref.float(), x_in_reshaped.float())
        x_layer_ref = torch.bmm(h_pre_ref.float(), x_in_reshaped.float())

        x_res_ref = x_res_ref.reshape(batch, n_expand * c_x)

        x_res_ref = x_res_ref.bfloat16()
        x_layer_ref = x_layer_ref.bfloat16()
        return x_res_ref, x_layer_ref


def main():
    batch = 1
    n_expand = 4
    c_x = 1280
    dtype = torch.bfloat16

    mhc_pre_benchmark = ManifoldConstrainedHyperConnectionPreBenchmark(batch, n_expand, c_x, dtype)
    op = ManifoldConstrainedHyperConnectionPreOp(batch, n_expand, c_x, dtype)
    input = mhc_pre_benchmark.gen_inputs()

    mhc_pre_benchmark.profile(op, *input)


if __name__ == "__main__":
    main()
