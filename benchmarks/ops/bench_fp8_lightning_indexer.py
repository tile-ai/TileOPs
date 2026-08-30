"""Benchmark for the FP8 lightning indexer op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts
come from the op's ``eval_roofline()`` via :class:`ManifestBenchmark`.

The reference materializes the ``[batch, heads, seq_len, seq_len_kv]`` scores the op
folds into its matmul epilogue, and peaks near 104 GB on these rows. A device that
cannot hold that fails in the reference rather than in the op.
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


_SHAPE_KEYS = (
    "batch",
    "seq_len",
    "heads",
    "index_dim",
    "seq_len_kv",
    "kv_group",
    "clean_logits",
)


def _one_row_per_shape(workloads: list[dict]) -> list[dict]:
    """The rows this bench measures: one per shape.

    ``FP8LightningIndexerWorkload.gen_inputs`` emits bf16 and the op quantizes
    inside, so two rows differing only in ``dtypes`` are one measurement.
    """
    seen, rows = set(), []
    for w in workloads:
        shape = tuple(str(w[key]) for key in _SHAPE_KEYS)
        if shape in seen:
            continue
        seen.add(shape)
        rows.append(w)
    return rows


@pytest.mark.parametrize(
    "batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits",
    workload_params(
        _one_row_per_shape(load_workloads(FP8LightningIndexerFwdOp)),
        fields(*_SHAPE_KEYS),
        smoke_first=True,
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
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
