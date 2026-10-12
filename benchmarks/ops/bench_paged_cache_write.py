"""Benchmark the TileOPs paged KV cache write, one case per manifest call, against vLLM and the reference."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import VLLM_TAG, private_inputs, vllm_op
from tileops.ops import PagedKVCacheWriteFwdOp


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

    def run(k, v, k_pages, v_pages, slots, k_scale, v_scale):
        write(
            k,
            v,
            k_pages,
            v_pages,
            slots,
            "fp8" if quantized else "auto",
            k_scale if quantized else unit,
            v_scale if quantized else unit,
        )
        return k_pages, v_pages

    # Both pools are written in place, so the adapter runs on copies reset every round.
    return private_inputs(run, inputs, 2, 3)


@pytest.mark.parametrize("case", bench.cases(PagedKVCacheWriteFwdOp), ids=lambda case: case.id)
def test_paged_kv_cache_write_bench(case) -> None:
    inputs = case.inputs
    op = PagedKVCacheWriteFwdOp(**case.arguments)
    # Every implementation writes k and v into k_pages and v_pages.
    implementations = {
        VLLM_TAG: _vllm_write(inputs),
        # The reference clones both pools and returns the clones, so it writes nothing
        # it was handed and needs no private copies; it is the oracle for this case.
        "torch-ref": case.reference,
    }
    bench.Runner(op, case).compare(implementations)
