"""Benchmark for the FP8 lightning indexer op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts
come from the op's ``eval_roofline()`` via :class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import FP8LightningIndexerOp
from workloads.fp8_lightning_indexer import FP8LightningIndexerWorkload

# Autotuning and the kernel-config override are bench-run policy, not
# workload properties; manifest workloads do not carry them.
_TUNE = False
_CONFIG = None


class _FP8LightningIndexerBaseline(FP8LightningIndexerWorkload):
    """Adds baseline torch_baseline for benchmark profiling."""

    def torch_baseline(self, q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                    cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> tuple[torch.Tensor]:
        k = kv
        q = q.float()
        k = k.float()
        batch, seq_len, heads, index_dim = q.shape
        seq_len_kv = self.seq_len_kv
        kv_group = self.kv_group
        heads_per_group = heads // kv_group

        k = k.view(batch, seq_len_kv, kv_group, index_dim)
        q = q.view(batch, seq_len, kv_group, heads_per_group, index_dim)

        mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum("bsghd,bngd->bghsn", q, k)
        weights = weights.view(seq_len, kv_group, heads_per_group)
        weights = weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(-1)
        score = score.relu() * weights
        logits = score.sum(dim=2)
        logits = logits.permute(0, 2, 3, 1)
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1)
        logits = logits.masked_fill(~mask_expanded, float("-inf"))
        return (logits,)


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
    test = _FP8LightningIndexerBaseline(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                        clean_logits, _CONFIG)
    inputs = test.gen_inputs()

    op = FP8LightningIndexerOp(clean_logits=clean_logits, config=_CONFIG, tune=_TUNE)
    bm = ManifestBenchmark(_FP8_LIGHTNING_INDEXER_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.torch_baseline, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
