"""Benchmark for the FP8 lightning indexer op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts
come from the op's ``eval_roofline()`` via :class:`ManifestBenchmark`.
"""

import pytest

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import FP8LightningIndexerOp
from workloads.fp8_lightning_indexer import FP8LightningIndexerWorkload

# Autotuning and the kernel-config override are bench-run policy, not
# workload properties; manifest workloads do not carry them.
_TUNE = False
_CONFIG = None


_FP8_LIGHTNING_INDEXER_OP = "FP8LightningIndexerOp"

_SHAPE_KEYS = (
    "batch", "seq_len", "heads", "index_dim", "seq_len_kv", "kv_group", "clean_logits",
)
def _indexer_params() -> list:
    """Params from manifest workloads, deduped on shape.

    ``FP8LightningIndexerWorkload.gen_inputs`` emits bf16 and quantizes inside
    the op, so workloads differing only in ``dtypes`` are one measurement.
    """
    seen, params = set(), []
    for w in load_workloads(_FP8_LIGHTNING_INDEXER_OP):
        args = tuple(w[k] for k in _SHAPE_KEYS)
        if args in seen:
            continue
        seen.add(args)
        params.append(pytest.param(
            *args, id=w["label"],
            marks=pytest.mark.smoke if not params else pytest.mark.full))
    return params


@pytest.mark.parametrize(
    "batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits",
    _indexer_params(),
)
def test_fp8_lightning_indexer_bench(batch: int, seq_len: int, heads: int, index_dim: int,
                                     seq_len_kv: int, kv_group: int,
                                     clean_logits: bool) -> None:
    test = FP8LightningIndexerWorkload(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                       clean_logits, _CONFIG)
    inputs = test.gen_inputs()

    op = FP8LightningIndexerOp(clean_logits=clean_logits, config=_CONFIG, tune=_TUNE)
    bm = ManifestBenchmark(_FP8_LIGHTNING_INDEXER_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
