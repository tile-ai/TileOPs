from typing import Tuple

import torch

from workloads.base import WorkloadBase


class FFTTest(WorkloadBase):

    def __init__(self, n: int, dtype: torch.dtype, batch_shape: tuple = ()):
        self.n = n
        self.dtype = dtype
        self.batch_shape = batch_shape

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        x = torch.randn(*self.batch_shape, self.n, device='cuda', dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=-1)
