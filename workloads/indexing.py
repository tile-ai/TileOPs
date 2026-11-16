import functools

import torch

from workloads.device import run_device
from workloads.numerics import Exact
from workloads.workload_base import WorkloadBase


class ConcatWorkload(WorkloadBase):
    def __init__(
        self, shapes: list[tuple], dim: int, dtypes: list[torch.dtype], strided: bool = False
    ):
        self.shapes = [tuple(s) for s in shapes]
        self.dim = dim
        self.dtypes = list(dtypes)
        # Each element a transposed view of storage laid out with its last two axes swapped.
        self.strided = strided

    def gen_inputs(self) -> tuple[list[torch.Tensor]]:
        tensors = []
        for shape, dtype in zip(self.shapes, self.dtypes, strict=True):
            if self.strided:
                stored = (*shape[:-2], shape[-1], shape[-2])
                tensors.append(torch.randn(stored, device=run_device()).to(dtype).transpose(-1, -2))
            else:
                tensors.append(torch.randn(shape, device=run_device()).to(dtype))
        return (tensors,)

    def ref_program(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Each element cast to the promoted dtype, then joined along ``dim``."""
        dtype = functools.reduce(torch.promote_types, (t.dtype for t in tensors))
        return torch.cat([t.to(dtype) for t in tensors], self.dim)

    def torch_program(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, self.dim)

    def verification(self, *inputs):
        # A copy and an exact widening: every element is bit-identical to the reference.
        return Exact(rtol=0, atol=0)
