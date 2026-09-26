"""Benchmark for the FP8 lightning indexer op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts
come from the op's ``eval_roofline()`` via :class:`ManifestBenchmark`.

The reference materializes the ``[batch, heads, seq_len, seq_len_kv]`` scores the op
folds into its matmul epilogue, and peaks near 104 GB on these rows. A device that
cannot hold that fails in the reference rather than in the op.
"""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import FP8LightningIndexerFwdOp
from workloads.fp8_lightning_indexer import FP8LightningIndexerCall


@pytest.mark.parametrize("call", manifest_calls(FP8LightningIndexerFwdOp))
def test_fp8_lightning_indexer_bench(call) -> None:
    test = FP8LightningIndexerCall(call)
    inputs = test.gen_inputs()

    op = FP8LightningIndexerFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
