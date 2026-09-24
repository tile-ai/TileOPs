import torch

from workloads.workload_base import WorkloadBase


class FFTWorkload(WorkloadBase):
    def __init__(self, n: int, dtype: torch.dtype, batch_shape: tuple = ()):
        self.n = n
        self.dtype = dtype
        self.batch_shape = batch_shape

    def gen_inputs(self, *, device: torch.device | str = "cuda") -> tuple[torch.Tensor]:
        x = torch.randn(*self.batch_shape, self.n, device=device, dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=-1)
