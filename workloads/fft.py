import torch

from workloads.workload_base import WorkloadBase


class FFTWorkload(WorkloadBase):

    def __init__(
        self,
        n: int,
        dtype: torch.dtype,
        batch_shape: tuple = (),
        layout: str = "contiguous",
    ):
        self.n = n
        self.dtype = dtype
        self.batch_shape = batch_shape
        self.layout = layout

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.layout == "strided":
            x = torch.randn(
                *self.batch_shape,
                self.n * 2,
                device="cuda",
                dtype=self.dtype,
            )[..., ::2]
        else:
            x = torch.randn(*self.batch_shape, self.n, device='cuda', dtype=self.dtype)
        if self.layout == "conjugate":
            x = x.conj()
        elif self.layout != "contiguous" and self.layout != "strided":
            raise ValueError(f"unsupported FFT workload layout: {self.layout}")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=-1)
