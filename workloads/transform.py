import math

import torch

from tileops.transform import HadamardTransformFwdOp
from workloads.device import run_device
from workloads.workload_base import WorkloadBase


class HadamardTransformWorkload(WorkloadBase):
    def __init__(self, batch_shape: tuple, n: int, dtype: torch.dtype, base_order: int = 1):
        self.batch_shape = tuple(batch_shape)
        self.n = n
        self.dtype = dtype
        self.base_order = base_order

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.batch_shape, self.n, device=run_device(), dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ kron(B, H_m) / sqrt(n)`` in float32, B the op's base matrix, H_m by butterflies."""
        base, m = self.base_order, self.n // self.base_order
        v = x.float().reshape(-1, base, m)
        h = 1
        while h < m:
            pairs = v.reshape(-1, base, m // (2 * h), 2, h)
            a, b = pairs[..., 0, :], pairs[..., 1, :]
            v = torch.stack([a + b, a - b], dim=-2).reshape(-1, base, m)
            h *= 2
        if base > 1:
            v = torch.einsum(
                "rak,ac->rck", v, HadamardTransformFwdOp.base_matrix(base).to(v.device)
            )
        return (v / math.sqrt(self.n)).reshape(x.shape).to(x.dtype)
