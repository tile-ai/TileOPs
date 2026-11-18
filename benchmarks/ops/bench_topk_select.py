"""Benchmark for the top-k selector op.

Workload shapes, dtypes, and ``top_k`` come from the ops manifest; roofline
FLOP and byte counts come from the op's ``eval_roofline()`` via
:class:`benchmarks.api.Runner`.

The row is timed against torch eager, the same reference through inductor, and the
sparse-attention indexer top-k of FlashInfer and of vLLM. flag_gems' top-k selects over
the last dimension only, and this op selects over ``seq_len_kv``, dim 2 of 4.

Both libraries read one score row per query, which is ``seq_len_kv`` only while
``kv_group`` is 1, and fill a short window's slots with -1: a row filling with another
index has neither tag, and a window not starting at 0 has no vLLM tag. Their row metadata
(lengths, offsets, workspace, output buffer) is built once outside the timed call, as
their serving engines build it once a step for every layer.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    flashinfer_op,
    vllm_op,
)
from tileops.attention import TopKSelectFwdOp
from workloads.attention.topk_select import TopKSelectCall

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True

# The query rows per request vLLM's indexer serves as decode, 1 + its speculative tokens:
# one, or two under MTP. A longer request is prefill.
_VLLM_DECODE_ROWS = 2


def _fills_with_minus_one(
    workload: TopKSelectCall, starts: torch.Tensor, ends: torch.Tensor
) -> bool:
    """Whether -1 filling, the libraries' only one, gives this row: it fills with -1, or
    every window holds ``top_k`` keys and nothing is filled."""
    return workload.fill_value == -1 or bool((ends - starts >= workload.topk).all())


def _flashinfer_topk(workload: TopKSelectCall, starts: torch.Tensor, ends: torch.Tensor):
    """FlashInfer's indexer top-k, ``top_k_ragged_transform``, or None for a row it cannot
    serve.

    It selects ``[row_starts, row_starts + lengths)`` of each row and returns int32
    positions plus ``offsets``. Windows from 0 pass no ``row_starts``, which its cluster
    path, the fastest, requires.
    """
    if workload.kv_group != 1 or not _fills_with_minus_one(workload, starts, ends):
        return None

    transform = flashinfer_op("top_k_ragged_transform")
    rows, topk = workload.batch * workload.seq_len, workload.topk
    starts, ends = starts.reshape(rows), ends.reshape(rows)
    if bool((starts == 0).all()):
        offsets, lengths, windows = torch.zeros_like(starts), ends, {}
    else:
        offsets, lengths, windows = starts, ends - starts, {"row_starts": starts}

    def fn(logits, *_):
        scores = logits.reshape(rows, workload.seq_len_kv)
        indices = transform(scores, offsets, lengths, topk, **windows)
        return indices.reshape(workload.batch, workload.seq_len, 1, topk)

    return fn


def _vllm_topk(workload: TopKSelectCall, starts: torch.Tensor, ends: torch.Tensor):
    """vLLM's indexer top-k, picked as ``sparse_attn_indexer`` picks it, or None for a row
    it cannot serve.

    Prefill takes ``top_k_per_row_prefill``, whose positions are relative to the window
    start, so a window not starting at 0 has no tag. Decode takes ``cooperative_topk``
    for at most 32 rows on SM90 with 16-byte aligned rows, else ``persistent_topk``, both
    for top_k 512, 1024 or 2048, else ``top_k_per_row_decode``; each reads windows
    ``[0, seq_len)``, one per row here.
    """
    if (
        workload.kv_group != 1
        or not bool((starts == 0).all())
        or not _fills_with_minus_one(workload, starts, ends)
    ):
        return None

    rows, topk, width = workload.batch * workload.seq_len, workload.topk, workload.seq_len_kv
    ends = ends.reshape(rows)
    out = torch.empty(rows, topk, dtype=torch.int32, device=ends.device)
    if workload.seq_len > _VLLM_DECODE_ROWS:
        prefill = vllm_op("top_k_per_row_prefill")
        starts = starts.reshape(rows)

        def select(scores):
            prefill(scores, starts, ends, out, rows, width, 1, topk)

    elif topk in (512, 1024, 2048):
        workspace = torch.empty(
            vllm_op("RADIX_TOPK_WORKSPACE_SIZE", "model_executor.layers.sparse_attn_indexer"),
            dtype=torch.uint8,
            device=ends.device,
        )
        major, minor = torch.cuda.get_device_capability(ends.device)
        if rows <= 32 and width % 4 == 0 and (major, minor) >= (9, 0) and major != 12:
            cooperative, longest = torch.ops._C.cooperative_topk, int(ends.max())

            def select(scores):
                cooperative(scores, ends, out, workspace, topk, longest)

        else:
            persistent = torch.ops._C.persistent_topk

            def select(scores):
                persistent(scores, ends, out, workspace, topk, width)

    else:
        decode = vllm_op("top_k_per_row_decode")

        def select(scores):
            decode(scores, 1, ends, out, rows, width, 1, topk)

    def fn(logits, *_):
        select(logits.reshape(rows, width))
        return out.reshape(workload.batch, workload.seq_len, 1, topk)

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
    for tag, adapter in ((FLASHINFER_TAG, _flashinfer_topk), (VLLM_TAG, _vllm_topk)):
        fn = adapter(case.workload, inputs[1], inputs[2])
        if fn is not None:
            implementations[tag] = fn

    bench.Runner(op, case).compare(implementations)
