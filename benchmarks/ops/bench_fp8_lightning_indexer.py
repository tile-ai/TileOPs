"""Benchmark for the FP8 lightning indexer op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts
come from the op's ``eval_roofline()`` via :class:`ManifestBenchmark`.
"""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, fields, workload_params
from tileops.manifest import load_workloads
from tileops.ops import FP8LightningIndexerFwdOp
from workloads.fp8_lightning_indexer import FP8LightningIndexerWorkload

# Autotuning and the kernel-config override are bench-run policy, not
# workload properties; manifest workloads do not carry them.
_TUNE = False
_CONFIG = None


_FP8_LIGHTNING_INDEXER_OP = "FP8LightningIndexerFwdOp"

_SHAPE_KEYS = (
    "batch",
    "seq_len",
    "heads",
    "index_dim",
    "seq_len_kv",
    "kv_group",
    "clean_logits",
)


@pytest.mark.parametrize(
    "batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits",
    workload_params(
        load_workloads(_FP8_LIGHTNING_INDEXER_OP),
        fields(*_SHAPE_KEYS),
        smoke_first=True,
        # gen_inputs emits bf16 and the op quantizes inside, so the dtype rows
        # of one shape are one measurement.
        dedupe_on=_SHAPE_KEYS,
    ),
)
def test_fp8_lightning_indexer_bench(
    batch: int,
    seq_len: int,
    heads: int,
    index_dim: int,
    seq_len_kv: int,
    kv_group: int,
    clean_logits: bool,
) -> None:
    test = FP8LightningIndexerWorkload(
        batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, _CONFIG
    )
    inputs = test.gen_inputs()

    op = FP8LightningIndexerFwdOp(clean_logits=clean_logits, config=_CONFIG, tune=_TUNE)
    bm = ManifestBenchmark(_FP8_LIGHTNING_INDEXER_OP, op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )
