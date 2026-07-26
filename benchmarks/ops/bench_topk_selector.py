"""Benchmark for the top-k selector op.

Workload shapes, dtypes, and ``topk`` come from the ops manifest; roofline
FLOP and byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import TopkSelectorOp
from workloads.topk_selector import TopkSelectorTest

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


def _workload_params(workloads: list, keys: tuple) -> list:
    """Turn manifest workload dicts into pytest params.

    First workload is marked ``smoke``, the rest ``full``. Keys ending in
    ``dtype`` are resolved to ``torch.dtype`` values.
    """
    params = []
    for i, w in enumerate(workloads):
        args = [getattr(torch, w[k]) if k.endswith("dtype") else w[k] for k in keys]
        params.append(
            pytest.param(
                *args,
                marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                id=w["label"],
            )
        )
    return params


class _TopkSelectorTestBaseline(TopkSelectorTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, index_score: torch.Tensor, starts: torch.Tensor,
                    ends: torch.Tensor) -> torch.Tensor:
        # index_score: (batch, seq_len, seq_len_kv, kv_group); topk over seq_len_kv (dim=2)
        indexes_ref = torch.topk(index_score, self.topk, dim=2)[1]
        # Match kernel/output layout: (batch, seq_len, kv_group, topk)
        return indexes_ref.permute(0, 1, 3, 2)


_TOPK_SELECTOR_OP = "TopkSelectorOp"
_TOPK_SELECTOR_PARAMS = _workload_params(
    load_workloads(_TOPK_SELECTOR_OP),
    ("batch", "seq_len", "seq_len_kv", "kv_group", "topk", "in_dtype", "out_dtype"),
)


@pytest.mark.parametrize(
    "batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype",
    _TOPK_SELECTOR_PARAMS,
)
def test_topk_selector_bench(batch: int, seq_len: int, seq_len_kv: int, kv_group: int, topk: int,
                             in_dtype: torch.dtype, out_dtype: torch.dtype) -> None:
    test = _TopkSelectorTestBaseline(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype,
                                     out_dtype)
    inputs = test.gen_inputs()

    op = TopkSelectorOp(topk=topk, tune=_TUNE)
    bm = ManifestBenchmark(_TOPK_SELECTOR_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
