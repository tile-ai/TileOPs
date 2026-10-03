"""Benchmark for the top-k selector op.

Workload shapes, dtypes, and ``topk`` come from the ops manifest; roofline
FLOP and byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

The row is timed against torch eager, the same reference through inductor, and
FlashInfer's top-k. flag_gems' top-k selects over the last dimension only, and this
op selects over ``seq_len_kv``, dim 2 of 4; FlashInfer's carries the same restriction
but a single ``kv_group`` makes that dimension the last one.
"""

import pytest
import torch

from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flashinfer_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import TopKSelectFwdOp
from workloads.attention.topk_select import TopkSelectorCall

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


def _flashinfer_topk(workload: TopkSelectorCall, starts: torch.Tensor, ends: torch.Tensor):
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


def _assert_selects_same_scores(fn, reference, *inputs: torch.Tensor) -> None:
    """Check a baseline selects the same scores, not the same order among ties.

    Two exact top-k implementations disagree on which index they keep where scores
    tie, so comparing index tensors would reject a correct baseline.

    Raises:
        AssertionError: When the selected scores differ.
    """
    flat = inputs[0].squeeze(-1).flatten(0, 1)

    def selected(indices: torch.Tensor) -> torch.Tensor:
        gathered = torch.gather(flat, 1, indices.reshape(flat.shape[0], -1).long())
        return torch.sort(gathered, dim=-1)[0]

    torch.testing.assert_close(selected(fn(*inputs)), selected(reference(*inputs)))


@pytest.mark.parametrize("call", manifest_calls(TopKSelectFwdOp))
def test_topk_select_bench(call) -> None:
    workload = TopkSelectorCall(call)
    inputs = workload.gen_inputs()

    op = TopKSelectFwdOp(**workload.arguments(), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)

    functors = {
        "tileops": op,
        "torch": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
    }
    flashinfer_fn = _flashinfer_topk(workload, inputs[1], inputs[2])
    if flashinfer_fn is not None:
        _assert_selects_same_scores(flashinfer_fn, workload.ref_program, *inputs)
        functors[FLASHINFER_TAG] = flashinfer_fn

    bm.compare(functors, *inputs)
