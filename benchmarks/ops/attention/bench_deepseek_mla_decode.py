import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek import MlaDecodeCall


@pytest.mark.parametrize("call", manifest_calls(MultiHeadLatentAttentionDecodeWithKVCacheFwdOp))
def test_mla_decode_bench(call) -> None:
    test = MlaDecodeCall(call)
    inputs = test.gen_inputs()

    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(**test.arguments(), tune=True)
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
