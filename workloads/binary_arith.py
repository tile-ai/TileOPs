import torch

from workloads.workload_base import WorkloadBase


class AddSameShapeTest(WorkloadBase):

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b
