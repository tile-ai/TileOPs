from typing import Tuple

import torch
import torch.nn.functional as F

from benchmarks.benchmark import Benchmark
from top.ops import Fp8QuantOp


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
        input_tensor_memory = self.seq_len_kv * self.index_dim * self.in_dtype.itemsize
        return input_tensor_memory

    def gen_inputs(self) -> torch.Tensor:
        input_tensor = torch.randn(
            self.seq_len_kv, self.index_dim, dtype=self.in_dtype, device="cuda")
        return input_tensor

    def ref_program(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        amax_value = torch.abs(input_tensor).amax(dim=1, keepdim=True).clamp(min=1e-4)
        scale_tensor = amax_value / 448.0
        output_tensor = torch.clamp(input_tensor / scale_tensor, min=-448.0, max=448.0)
        output_tensor = output_tensor.to(torch.float8_e4m3fn)
        return scale_tensor.squeeze(dim=1), output_tensor

    def __check_common(self,
                       *inputs: Tuple[torch.Tensor],
                       atol: float,
                       rtol: float,
                       op=None,
                       fn=None) -> None:
        """Common logic to check the correctness of the operation or function"""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        with torch.no_grad():
            outputs = fn(*inputs) if fn else op(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), "outputs and outputs_ref have different size"
        for i, (output, output_ref) in enumerate(zip(outputs, outputs_ref)):
            if output_ref is not None:
                output = output.to(torch.float32)
                output_ref = output_ref.to(torch.float32)
                cos_sim = F.cosine_similarity(output.flatten(), output_ref.flatten(), dim=0)
                cosine_threshold = 0.99
                assert cos_sim >= cosine_threshold, f"outputs[{i}] is not close to outputs_ref[{i}]. Cosine similarity: {cos_sim.item()}"

        print("All checks passed.✅")

    def check(self,
              op,
              *inputs: Tuple[torch.Tensor],
              atol: float = 1e-2,
              rtol: float = 1e-2) -> None:
        """Check the correctness of the operation"""
        self.__check_common(*inputs, atol=atol, rtol=rtol, op=op)

    def check_fn(self,
                 fn,
                 *inputs: Tuple[torch.Tensor],
                 atol: float = 1e-2,
                 rtol: float = 1e-2,
                 grad=False) -> None:
        """Check the correctness of the function/layer"""
        self.__check_common(*inputs, atol=atol, rtol=rtol, fn=fn)
