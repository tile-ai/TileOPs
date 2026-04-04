import torch
import torch.nn.functional as F

from workloads.base import WorkloadBase


class LogSoftmaxTest(WorkloadBase):
    """TestBase adapter for spec-interface LogSoftmaxOp."""

    def __init__(self, shape: tuple[int, ...], dim: int, dtype: torch.dtype):
        self.shape = shape
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x.float(), dim=self.dim).to(x.dtype)


class LogSumExpTest(WorkloadBase):
    """TestBase adapter for spec-interface LogSumExpOp."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dim: int,
        dtype: torch.dtype,
        keepdim: bool = False,
    ):
        self.shape = shape
        self.dim = dim
        self.dtype = dtype
        self.keepdim = keepdim

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=self.dim, keepdim=self.keepdim).to(
            x.dtype
        )


class SoftmaxTest(WorkloadBase):
    """TestBase adapter for spec-interface SoftmaxOp."""

    def __init__(self, shape: tuple[int, ...], dim: int, dtype: torch.dtype):
        self.shape = shape
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x.float(), dim=self.dim).to(x.dtype)
