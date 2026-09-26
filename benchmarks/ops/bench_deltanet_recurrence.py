"""Benchmark: TileOPs DeltaNet decode, one case per manifest call, against its torch reference."""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import DeltaNetDecodeFwdOp
from workloads.linear_attention import DeltaNetDecodeCall


@pytest.mark.parametrize("call", manifest_calls(DeltaNetDecodeFwdOp))
def test_deltanet_decode_bench(call) -> None:
    test = DeltaNetDecodeCall(call)
    inputs = test.gen_inputs()
    op = DeltaNetDecodeFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    bm.compare(
        {
            "tileops": op,
            "torch": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
