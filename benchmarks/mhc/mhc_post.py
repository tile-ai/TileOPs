from typing import Tuple

import torch

from benchmarks.benchmark import Benchmark
from top.ops import ManifoldConstrainedHyperConnectionPostOp


class ManifoldConstrainedHyperConnectionPostBenchmark(Benchmark):

    op_type = ManifoldConstrainedHyperConnectionPostOp

    def __init__(self, batch: int, n_expand: int, c_x: int, dtype: torch.dtype):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype

    @property
    def total_flops(self) -> float:
        flops = 2 * self.batch * (
            self.n_expand * self.n_expand * self.c_x + self.n_expand * self.c_x)
        return flops

    @property
    def total_memory(self) -> float:
        return (self.n_expand * 2 + 1) * self.c_x

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_layer_out = torch.randn([batch, c_x], device="cuda", dtype=torch.bfloat16)
        h_post = torch.randn([batch, n_expand], device="cuda", dtype=torch.float32)
        x_res = torch.randn([batch, n_expand * c_x], device="cuda", dtype=torch.bfloat16)
        return x_layer_out, h_post, x_res

    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_out_ref = torch.zeros([batch, n_expand * c_x], device="cuda", dtype=torch.bfloat16)
        for i in range(batch):
            x_out_ref[i, :] = (h_post[i, :].reshape(n_expand, 1).float()
                               @ x_layer_out[i, :].reshape(1, c_x).float()).reshape(
                                   n_expand * c_x) + x_res[i, :].float()
        return x_out_ref


def main():
    batch = 1
    n_expand = 4
    c_x = 1280
    dtype = torch.bfloat16

    mhc_post_benchmark = ManifoldConstrainedHyperConnectionPostBenchmark(
        batch, n_expand, c_x, dtype)
    op = ManifoldConstrainedHyperConnectionPostOp(batch, n_expand, c_x, dtype)
    input = mhc_post_benchmark.gen_inputs()

    mhc_post_benchmark.profile(op, *input)


if __name__ == "__main__":
    main()
