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
from benchmarks.verification import Custom
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
        functors[FLASHINFER_TAG] = flashinfer_fn

    def validate(got, expected):
        assert got.shape == expected.shape and got.dtype == expected.dtype
        scores, starts, ends = inputs
        padding = got == scores.shape[2]
        assert (
            padding | ((got >= starts[:, :, None, None]) & (got < ends[:, :, None, None]))
        ).all()
        assert torch.equal(padding.sum(-1), (expected == scores.shape[2]).sum(-1))
        ordered = got.sort(-1).values
        assert (
            (ordered[..., 1:] != ordered[..., :-1]) | (ordered[..., 1:] == scores.shape[2])
        ).all(), "duplicate selected index"
        scores = torch.nn.functional.pad(scores.movedim(2, -1), (0, 1), value=-float("inf"))
        torch.testing.assert_close(
            scores.gather(-1, got.long()).sort(-1).values,
            scores.gather(-1, expected.long()).sort(-1).values,
            rtol=0,
            atol=0,
        )

    bm.compare(
        functors,
        *inputs,
        evidence=dict.fromkeys(
            functors,
            Custom(
                validate,
                "same top-k scores with distinct in-window indices; tie order is unspecified",
            ),
        ),
    )
