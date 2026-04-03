"""Benchmark for SharedExpertMLPKernel vs PyTorch MLP."""

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.test_base import FixtureBase, TestBase
from tileops.kernels.moe import SharedExpertMLPKernel


class SharedMLPBenchTest(TestBase):
    def __init__(self, num_tokens, hidden_size, ffn_size, dtype):
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype

    def gen_inputs(self):
        device = torch.device("cuda")
        hidden = torch.randn(self.num_tokens, self.hidden_size, dtype=self.dtype, device=device)
        w_gate_up = torch.randn(self.ffn_size * 2, self.hidden_size, dtype=self.dtype, device=device)
        w_down = torch.randn(self.hidden_size, self.ffn_size, dtype=self.dtype, device=device)
        return hidden, w_gate_up, w_down

    def ref_program(self, hidden, w_gate_up, w_down):
        gate = torch.nn.functional.linear(hidden, w_gate_up[:self.ffn_size])
        up = torch.nn.functional.linear(hidden, w_gate_up[self.ffn_size:])
        gate_up = torch.nn.functional.silu(gate) * up
        return torch.nn.functional.linear(gate_up, w_down)


class SharedMLPBenchFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, hidden_size, ffn_size, dtype",
            [
                pytest.param(512, 2048, 8192, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(2048, 2048, 8192, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(4096, 2048, 8192, torch.bfloat16, marks=pytest.mark.full),
            ],
        )
    ]


class SharedMLPBenchmark(BenchmarkBase):
    def calculate_flops(self):
        t = self.test
        return 2 * t.num_tokens * t.hidden_size * t.ffn_size * 3  # gate + up + down

    def calculate_memory(self):
        t = self.test
        elem = 2  # bf16
        return elem * (t.num_tokens * t.hidden_size + 3 * t.ffn_size * t.hidden_size + 3 * t.num_tokens * t.ffn_size)


@SharedMLPBenchFixture
def test_shared_mlp_bench(num_tokens, hidden_size, ffn_size, dtype):
    test = SharedMLPBenchTest(num_tokens, hidden_size, ffn_size, dtype)
    bm = SharedMLPBenchmark(test)
    hidden, w_gate_up, w_down = test.gen_inputs()

    # TileLang kernel
    kernel = SharedExpertMLPKernel(num_tokens=num_tokens, hidden_size=hidden_size,
                                   ffn_size=ffn_size, dtype=dtype)
    kernel(hidden, w_gate_up, w_down)  # warmup
    torch.cuda.synchronize()

    result = bm.profile(kernel, hidden, w_gate_up, w_down)
    BenchmarkReport.record(kernel, locals(), result, tag="tileops")

    # PyTorch baseline
    def pytorch_fn(hidden, w_gate_up, w_down):
        gate = torch.nn.functional.linear(hidden, w_gate_up[:ffn_size])
        up = torch.nn.functional.linear(hidden, w_gate_up[ffn_size:])
        gate_up = torch.nn.functional.silu(gate) * up
        return torch.nn.functional.linear(gate_up, w_down)

    pytorch_fn(hidden, w_gate_up, w_down)  # warmup
    torch.cuda.synchronize()

    result_torch = bm.profile(pytorch_fn, hidden, w_gate_up, w_down)
    BenchmarkReport.record(kernel, locals(), result_torch, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])
