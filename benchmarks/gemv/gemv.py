from typing import Tuple

import torch

from benchmarks.benchmark import Benchmark
from top.ops import GemvOp


class GemvBenchmark(Benchmark):

    op_type = GemvOp

    def __init__(self, n: int, k: int, dtype: torch.dtype):
        self.n = n
        self.k = k
        self.dtype = dtype

    @property
    def total_flops(self) -> float:
        return 2.0 * self.n * self.k

    @property
    def total_memory(self) -> int:
        return (self.k + self.k * self.n + self.n) * self.dtype.itemsize

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        shape_a = (self.k,)
        a = torch.randn(*shape_a, device='cuda', dtype=self.dtype)
        shape_b = (self.n, self.k)
        b = torch.randn(*shape_b, device='cuda', dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # return torch.mv(b, a)
        return b @ a

    def baseline_profile(self, *inputs, warmup=100, rep=10, device="cuda:0") -> None:
        return super().baseline_profile(
            self.ref_program, *inputs, backend="torch", warmup=warmup, rep=rep, device=device)
