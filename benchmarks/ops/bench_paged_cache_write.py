"""Benchmark paged KV cache writes against vLLM, FlashInfer and the shared reference."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    VLLM_TAG,
    flashinfer_op,
    private_inputs,
    vllm_op,
)
from tileops.attention import PagedKVCacheWriteFwdOp


def _vllm_write(inputs):
    """vLLM's fused cache write, on the same slots and divisors.

    ``reshape_and_cache_flash`` takes the pool in the layout the manifest declares,
    ``[num_blocks, block_size, heads_kv, dim]``, and reads the slot as
    ``block * block_size + offset``, so both caches pass through unreshaped. It stores
    each row divided by the scale it is handed, which is this op's contract, so the
    divisors pass through too. The library is a selected baseline: an image without it
    fails this row rather than dropping the tag.
    """
    write = vllm_op("reshape_and_cache_flash")
    quantized = inputs[2].dtype == torch.float8_e4m3fn
    # The unquantized API still requires scale tensors. Prepare them outside timing.
    unit = None if quantized else torch.ones(1, dtype=torch.float32, device=inputs[0].device)

    def run(k, v, k_cache, v_cache, slots, k_scale, v_scale):
        write(
            k,
            v,
            k_cache,
            v_cache,
            slots,
            "fp8" if quantized else "auto",
            k_scale if quantized else unit,
            v_scale if quantized else unit,
        )
        return k_cache, v_cache

    # Both pools are written in place, so the adapter runs on copies reset every round.
    return private_inputs(run, inputs, 2, 3)


def _flashinfer_write(inputs):
    """Represent each physical slot as one request's single page and append position.

    This adapter covers same-dtype 16-bit writes with no skipped tokens. Request
    metadata is prepared outside timing; K/V data and the page pools pass through.
    """
    write = flashinfer_op("append_paged_kv_cache")
    slots = inputs[4]
    page_size = inputs[2].shape[1]
    batch_indices = torch.arange(slots.numel(), dtype=torch.int32, device=slots.device)
    indptr = torch.arange(slots.numel() + 1, dtype=torch.int32, device=slots.device)
    indices = (slots // page_size).to(torch.int32)
    positions = (slots % page_size).to(torch.int32)
    last_page_len = positions + 1

    def run(k, v, k_cache, v_cache, slots, k_scale, v_scale):
        write(k, v, batch_indices, positions, (k_cache, v_cache), indices, indptr, last_page_len)
        return k_cache, v_cache

    return private_inputs(run, inputs, 2, 3)


@pytest.mark.parametrize("case", bench.cases(PagedKVCacheWriteFwdOp), ids=lambda case: case.id)
def test_paged_kv_cache_write_bench(case) -> None:
    inputs = case.inputs
    op = PagedKVCacheWriteFwdOp(**case.arguments)
    # Every implementation writes k and v into k_cache and v_cache.
    implementations = {
        VLLM_TAG: _vllm_write(inputs),
        # The reference clones both pools and returns the clones, so it writes nothing
        # it was handed and needs no private copies; it is the oracle for this case.
        "torch-ref": case.reference,
    }
    # FlashInfer has no independent quantization scales or negative-slot skip input.
    if inputs[2].dtype != torch.float8_e4m3fn and bool((inputs[4] >= 0).all()):
        implementations[FLASHINFER_TAG] = _flashinfer_write(inputs)
    bench.Runner(op, case).compare(implementations)
