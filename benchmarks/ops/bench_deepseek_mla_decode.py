"""Benchmark TileOPs DeepSeek multi-head latent attention (MLA) decode, one case per manifest call, against its torch reference."""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from workloads.deepseek_attention import MlaDecodeCall


@pytest.mark.parametrize("call", manifest_calls(MultiHeadLatentAttentionDecodeWithKVCacheFwdOp))
def test_mla_decode_bench(call) -> None:
    workload = MlaDecodeCall(call)
    inputs = workload.gen_inputs()

    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(**workload.arguments(), tune=True)
    bm = ManifestBenchmark(op, workload)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )
