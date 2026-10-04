import torch

from workloads.device import run_device
from workloads.workload_base import WorkloadBase


class FFTWorkload(WorkloadBase):
    def __init__(self, n: int, dtype: torch.dtype, batch_shape: tuple = ()):
        self.n = n
        self.dtype = dtype
        self.batch_shape = batch_shape

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.batch_shape, self.n, device=run_device(), dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=-1)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        single = inputs[0].dtype == torch.complex64
        rtol = 1e-4 if single else 1e-8
        atol = rtol
        if self.n >= 1 << 15:
            # Decomposed FFT output magnitude and absolute roundoff grow with sqrt(n).
            scale = (self.n / (1 << 20)) ** 0.5
            atol = (6e-3 if single else 2e-11) * scale
        return Exact(atol=atol, rtol=rtol)
