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


def _cosine_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using cosine similarity (fp8 quantization needs looser checks)."""
    output = output.to(torch.float32)
    output_ref = output_ref.to(torch.float32)
    cos_sim = F.cosine_similarity(output.flatten(), output_ref.flatten(), dim=0)
    assert cos_sim >= 0.99, \
        f"Cosine similarity too low: {cos_sim.item()}"


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


@Fp8QuantFixture
def test_fp8_quant_op(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype,
                      tune: bool) -> None:
    test = Fp8QuantTest(seq_len_kv, index_dim, in_dtype)
    op = Fp8QuantOp(seq_len_kv=seq_len_kv, index_dim=index_dim, in_dtype=in_dtype, tune=tune)
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
