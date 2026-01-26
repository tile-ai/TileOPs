from typing import Tuple

import torch

from benchmarks.benchmark import Benchmark
from top.ops import Fp8QuantOp

class Fp8QuantBenchmark(Benchmark):
    op_type = Fp8QuantOp

    def __init__(
        self,
        seq_len_kv,
        index_dim,
        in_dtype
    ) -> None:
        self.seq_len_kv = seq_len_kv
        self.index_dim = index_dim
        self.in_dtype = in_dtype


    @property
    def total_flops(self) -> float:
        return 2 * self.seq_len_kv * index_dim + self.seq_len_kv + 4 * self.seq_len_kv * self.index_dim


    @property
    def total_memory(self) -> float:
        # input_tensor: seq_len_kv, index_dim
        input_tensor_memory = self.seq_len_kv * self.index_dim * torch.float16.itemsize
        return input_tensor_memory

    def gen_inputs(self) -> torch.Tensor:
        input_tensor = torch.randn(self.batch, self.seq_len, dtype=self.in_dtype, device="cuda")
        return input_tensor


    def ref_program(self, input_tensor: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor]:
        output_tensor = torch.empty((self.seq_len_kv, self.index_dim), torch.float8)
        scale_tensor = torch.empty((self.seq_len_kv), torch.float32)
        for i in range(self.seq_len_kv):
            scale_tensor[i] = torch.amax(input_tensor[i,:]) /448.0
            output_tensor[i,:] = input_tensor[i,:] / scale_tensor[i]
        return scale_tensor, output_tensor
