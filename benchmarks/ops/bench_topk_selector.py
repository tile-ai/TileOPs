"""Benchmark for the top-k selector op.

Workload shapes, dtypes, and ``topk`` come from the ops manifest; roofline
FLOP and byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

The row is timed against torch eager and the same reference through inductor.
flag_gems' top-k selects over the last dimension only, and this op selects over
``seq_len_kv``, dim 2 of 4.
"""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    workload_field_params,
)
from tileops.manifest import load_workloads
from tileops.ops import TopkSelectorFwdOp
from workloads.topk_selector import TopkSelectorWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


_TOPK_SELECTOR_OP = "TopkSelectorFwdOp"
_TOPK_SELECTOR_PARAMS = workload_field_params(
    load_workloads(_TOPK_SELECTOR_OP),
    ("batch", "seq_len", "seq_len_kv", "kv_group", "topk", "in_dtype", "out_dtype"),
)


@pytest.mark.parametrize(
    "batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype",
    _TOPK_SELECTOR_PARAMS,
)
def test_topk_selector_bench(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    kv_group: int,
    topk: int,
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    test = TopkSelectorWorkload(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)
    inputs = test.gen_inputs()

    op = TopkSelectorFwdOp(topk=topk, tune=_TUNE)
    bm = ManifestBenchmark(_TOPK_SELECTOR_OP, op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )
