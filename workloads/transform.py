import math
from dataclasses import dataclass

import torch

from tileops.transform import HadamardTransformFwdOp
from workloads.device import run_device
from workloads.workload_base import WorkloadBase


@dataclass(eq=False)
class HadamardTransformWorkload(WorkloadBase):
    batch_shape: tuple
    n: int
    dtype: torch.dtype
    base_order: int = 1

    def __post_init__(self) -> None:
        self.batch_shape = tuple(self.batch_shape)

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.batch_shape, self.n, device=run_device(), dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return hadamard_transform(x, self.base_order)


def hadamard_transform(x: torch.Tensor, base_order: int = 1) -> torch.Tensor:
    """``x @ kron(B, H_m) / sqrt(n)`` over the last axis in float32, rounded to ``x``'s dtype;
    B the op's base matrix of ``base_order``, H_m the Sylvester matrix by butterflies."""
    n = x.shape[-1]
    base, m = base_order, n // base_order
    v = x.float().reshape(-1, base, m)
    h = 1
    while h < m:
        pairs = v.reshape(-1, base, m // (2 * h), 2, h)
        a, b = pairs[..., 0, :], pairs[..., 1, :]
        v = torch.stack([a + b, a - b], dim=-2).reshape(-1, base, m)
        h *= 2
    if base > 1:
        v = torch.einsum("rak,ac->rck", v, HadamardTransformFwdOp.base_matrix(base).to(v.device))
    return (v / math.sqrt(n)).reshape(x.shape).to(x.dtype)
