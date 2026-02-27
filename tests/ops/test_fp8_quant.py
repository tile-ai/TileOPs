from typing import Tuple

import torch
import torch.nn.functional as F
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops import Fp8QuantOp


class Fp8QuantFixture(FixtureBase):
    PARAMS = [
        ("seq_len_kv, index_dim, in_dtype, tune", [
            (8192, 64, torch.float16, False),
            (8192, 64, torch.bfloat16, False),
            (4096, 128, torch.float32, False),
            (16384, 32, torch.float32, False),
        ]),
    ]


class Fp8QuantTest(TestBase):

    def __init__(self, seq_len_kv: int, index_dim: int, in_dtype: torch.dtype):
        self.seq_len_kv = seq_len_kv
        self.index_dim = index_dim
        self.in_dtype = in_dtype

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        input_tensor = torch.randn(
            self.seq_len_kv, self.index_dim, dtype=self.in_dtype, device="cuda")
        return (input_tensor,)

    def ref_program(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        amax_value = torch.abs(input_tensor).amax(dim=1, keepdim=True).clamp(min=1e-4)
        scale_tensor = amax_value / 448.0
        output_tensor = torch.clamp(input_tensor / scale_tensor, min=-448.0, max=448.0)
        output_tensor = output_tensor.to(torch.float8_e4m3fn)
        return scale_tensor.squeeze(dim=1), output_tensor

    def check(self,
              op,
              *inputs: Tuple[torch.Tensor],
              atol: float = 1e-2,
              rtol: float = 1e-2) -> None:
        """Check using cosine similarity (fp8 quantization needs looser checks)."""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
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
        for i, (output, output_ref) in enumerate(
                zip(outputs, outputs_ref, strict=True)):
            if output_ref is not None:
                output = output.to(torch.float32)
                output_ref = output_ref.to(torch.float32)
                cos_sim = F.cosine_similarity(output.flatten(), output_ref.flatten(), dim=0)
                cosine_threshold = 0.99
                assert cos_sim >= cosine_threshold, \
                    f"outputs[{i}] is not close to outputs_ref[{i}]. Cosine similarity: {cos_sim.item()}"

        print(f"All checks passed for {op.__class__.__name__}.")

    def check_fn(self,
                 fn: callable,
                 *inputs: Tuple[torch.Tensor],
                 atol: float = 1e-2,
                 rtol: float = 1e-2,
                 grad: bool = False) -> None:
        """Check function using cosine similarity."""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        with torch.no_grad():
            outputs = fn(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), \
            f"outputs: {len(outputs)} and outputs_ref: {len(outputs_ref)} have different size"
        for i, (output, output_ref) in enumerate(
                zip(outputs, outputs_ref, strict=True)):
            if output_ref is not None:
                output = output.to(torch.float32)
                output_ref = output_ref.to(torch.float32)
                cos_sim = F.cosine_similarity(output.flatten(), output_ref.flatten(), dim=0)
                cosine_threshold = 0.99
                assert cos_sim >= cosine_threshold, \
                    f"outputs[{i}] is not close to outputs_ref[{i}]. Cosine similarity: {cos_sim.item()}"

        print(f"All checks passed for {fn.__class__.__name__}.")


@Fp8QuantFixture
def test_fp8_quant_op(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype,
                      tune: bool) -> None:
    test = Fp8QuantTest(seq_len_kv, index_dim, in_dtype)
    op = Fp8QuantOp(seq_len_kv=seq_len_kv, index_dim=index_dim, in_dtype=in_dtype, tune=tune)
    test.check(op, *test.gen_inputs())


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
