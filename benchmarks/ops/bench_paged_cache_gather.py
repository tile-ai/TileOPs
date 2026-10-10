"""Paged gather against the workload reference and both vLLM cache gather entry points."""

from math import prod

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import VLLM_TAG, private_inputs, vllm_op
from tileops.ops import PagedKVCacheGatherFwdOp


def _vllm_gather(inputs):
    """The fused vLLM gather accepts flattened MLA widths 320 and 576 only."""
    dst, cache, table = inputs[:3]
    if prod(cache.shape[2:]) not in (320, 576):
        return None
    gather = vllm_op("gather_and_maybe_dequant_cache")
    unit = torch.ones(1, dtype=torch.float32, device=dst.device)
    batch_ids = torch.arange(table.shape[0], dtype=torch.int32, device=dst.device)

    def run(dst, cache, table, cu, starts, scale):
        # This API requires a mapping absent from the public call; constructing it
        # from live offsets is part of the timed operation, not benchmark setup.
        token_to_seq = torch.repeat_interleave(
            batch_ids, cu[1:] - cu[:-1], output_size=dst.shape[0]
        )
        gather(
            cache.flatten(2),
            dst.flatten(1),
            table,
            cu,
            token_to_seq,
            dst.shape[0],
            "fp8_e4m3" if scale is not None else "auto",
            scale if scale is not None else unit,
            starts,
        )
        return dst

    return private_inputs(run, inputs, 0)


def _vllm_cp_gather(inputs):
    """CP gather copies identical dtypes; it has no scale or dequantized-output contract."""
    if inputs[5] is not None:
        return None
    gather = vllm_op("cp_gather_cache")

    def run(dst, cache, table, cu, starts, _scale):
        gather(cache, dst, table, cu, table.shape[0], starts)
        return dst

    return private_inputs(run, inputs, 0)


@pytest.mark.parametrize("case", bench.cases(PagedKVCacheGatherFwdOp), ids=lambda case: case.id)
def test_paged_kv_cache_gather_bench(case):
    implementations = {"torch-ref": case.reference}
    fused = _vllm_gather(case.inputs)
    if fused is not None:
        implementations[VLLM_TAG] = fused
    cp = _vllm_cp_gather(case.inputs)
    if cp is not None:
        implementations["vllm-cp-gather"] = cp
    bench.Runner(PagedKVCacheGatherFwdOp(**case.arguments), case).compare(implementations)
