"""Benchmark for the top-k selector op.

Workload shapes, dtypes, and ``topk`` come from the ops manifest; roofline
FLOP and byte counts come from the op's ``eval_roofline()`` via
:class:`benchmarks.api.Runner`.

The row is timed against torch eager, the same reference through inductor, and
FlashInfer's top-k. flag_gems' top-k selects over the last dimension only, and this
op selects over ``seq_len_kv``, dim 2 of 4; FlashInfer's carries the same restriction
but a single ``kv_group`` makes that dimension the last one.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flashinfer_op,
)
from tileops.ops import TopKSelectFwdOp
from workloads.attention.topk_select import TopKSelectCall

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


def _flashinfer_topk(workload: TopKSelectCall, starts: torch.Tensor, ends: torch.Tensor):
    """FlashInfer's top-k over the same scores, or None for a row it cannot serve.

    It selects over the last dimension, which is ``seq_len_kv`` only while
    ``kv_group`` is 1, and over the whole row, so a narrowed ``[start, end)`` is out.
    """
    if workload.kv_group != 1:
        return None
    if not (bool((starts == 0).all()) and bool((ends == workload.seq_len_kv).all())):
        return None

    top_k = flashinfer_op("top_k")
    rows, topk, out_dtype = workload.batch * workload.seq_len, workload.topk, workload.out_dtype

    def fn(index_score, *_):
        _, indices = top_k(index_score.squeeze(-1).reshape(rows, workload.seq_len_kv), topk)
        return indices.reshape(workload.batch, workload.seq_len, 1, topk).to(out_dtype)

    return fn


@pytest.mark.parametrize("case", bench.cases(TopKSelectFwdOp), ids=lambda case: case.id)
def test_topk_select_bench(case) -> None:
    inputs = case.inputs

    op = TopKSelectFwdOp(**case.arguments)
    if _TUNE:
        op.request_tune()

    implementations = {
        "torch": case.reference,
        TORCH_COMPILE_TAG: compiled_reference(case.reference),
    }
    flashinfer_fn = _flashinfer_topk(case.workload, inputs[1], inputs[2])
    if flashinfer_fn is not None:
        implementations[FLASHINFER_TAG] = flashinfer_fn

    bench.Runner(op, case).compare(implementations)
