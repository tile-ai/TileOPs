from typing import Tuple

import torch
import torch.nn.functional as F

from benchmarks.benchmark import Benchmark
from top.ops import Fp8QuantOp
from top.ops import Op


class Fp8QuantBenchmark(Benchmark):
    op_type = Fp8QuantOp

    def __init__(
        self,
        seq_len_kv,
        index_dim,
        in_dtype,
        tune: bool = False,
    ) -> None:
        self.seq_len_kv = seq_len_kv
        self.index_dim = index_dim
        self.in_dtype = in_dtype
        self.tune = tune

    @property
    def total_flops(self) -> float:
        return 2 * self.seq_len_kv * self.index_dim + self.seq_len_kv + 4 * self.seq_len_kv * self.index_dim

    @property
    def total_memory(self) -> float:
        # input_tensor: seq_len_kv, index_dim
        input_tensor_memory = self.seq_len_kv * self.index_dim * torch.float16.itemsize
        return input_tensor_memory

    def gen_inputs(self) -> torch.Tensor:
        input_tensor = torch.randn(
            self.seq_len_kv, self.index_dim, dtype=self.in_dtype, device="cuda")
        return input_tensor

    def ref_program(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tensor = torch.empty((self.seq_len_kv, self.index_dim),
                                    dtype=torch.torch.float32,
                                    device=input_tensor.device)
        scale_tensor = torch.empty((self.seq_len_kv),
                                   dtype=torch.float32,
                                   device=input_tensor.device)
        for i in range(self.seq_len_kv):
            amax_value = torch.amax(input_tensor[i, :]).clamp(min=1e-4)
            scale_tensor[i] = amax_value / 448.0
            output_tensor[i, :] = torch.clamp(
                input_tensor[i, :] / scale_tensor[i], min=-448.0, max=448.0)

        output_tensor = output_tensor.to(torch.float8_e4m3fn)
        return scale_tensor, output_tensor

    def check(self,
              op: Op,
              *inputs: Tuple[torch.Tensor],
              atol: float = 1e-2,
              rtol: float = 1e-2) -> None:
        """Check the correctness of the op"""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        with torch.no_grad():
            outputs = op(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), "outputs and outputs_ref have different size"
        print("outputs:", outputs)
        print("outputs_ref", outputs_ref)
        for i, (output, output_ref) in enumerate(zip(outputs, outputs_ref)):
            if output_ref is not None:  # skip checking for None placeholders in ref
                output = output.to(torch.float32)
                output_ref = output_ref.to(torch.float32)
                cos_sim = F.cosine_similarity(output.flatten(), output_ref.flatten(), dim=0)
                cosine_threshold = 0.98
                assert cos_sim >= cosine_threshold, f"outputs[{i}] is not close to outputs_ref[{i}]. Cosine similarity: {cos_sim.item()}"

        print(f"All checks passed for {op.__class__.__name__}.✅")
