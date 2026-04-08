from typing import Tuple

import torch

from workloads.base import WorkloadBase


class MHCPostTest(WorkloadBase):

    def __init__(self, batch: int, n_expand: int, c_x: int, dtype: torch.dtype):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_layer_out = torch.randn([batch, c_x], device="cuda", dtype=self.dtype)
        h_post = torch.randn([batch, n_expand], device="cuda", dtype=torch.float32)
        x_res = torch.randn([batch, n_expand * c_x], device="cuda", dtype=self.dtype)
        return x_layer_out, h_post, x_res
