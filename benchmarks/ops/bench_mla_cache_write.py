"""Benchmark MLA latent cache writes against vLLM, FlashInfer and the shared reference."""

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
from tileops.ops import MLAKVCacheWriteFwdOp


def _vllm_cache_args(inputs):
    """vLLM's cache dtype name and divisor: an FP8 cache passes its scale, any other a unit.

    The unquantized API still requires a scale tensor; it is prepared outside timing.
    """
    kv_cache, scale = inputs[2], inputs[4]
    if kv_cache.dtype == torch.float8_e4m3fn:
        return "fp8", scale
    return "auto", torch.ones(1, dtype=torch.float32, device=kv_cache.device)


def _cache_view(kv_cache):
    """vLLM addresses an FP8 cache through its bytes."""
    return kv_cache.view(torch.uint8) if kv_cache.dtype == torch.float8_e4m3fn else kv_cache


def _vllm_write(inputs):
    """vLLM's ``concat_and_cache_mla`` on the same slots and divisor.

    It takes the cache in the layout the manifest declares, ``[num_blocks, block_size,
    DC + PE]``, reads the slot as ``block * block_size + offset`` and skips ``-1``, and
    stores each value divided by the scale, so every tensor passes through. The library is a
    selected baseline: an image without it fails this row rather than dropping the tag.
    """
    write = vllm_op("concat_and_cache_mla")
    dtype_name, divisor = _vllm_cache_args(inputs)

    def run(kv_c, k_pe, kv_cache, slots, scale, positions, cos_sin_cache):
        write(kv_c, k_pe, _cache_view(kv_cache), slots, dtype_name, divisor)
        return kv_cache

    # The cache is written in place, so the adapter runs on a copy reset every round.
    return private_inputs(run, inputs, 2)


def _vllm_rope_fused_write(inputs, neox):
    """vLLM's ``concat_and_cache_mla_rope_fused``, which rotates ``k_pe`` as it writes.

    It also rotates a query; the query passed has no heads, so it does no query work. It
    rotates ``k_pe`` in place, and reads ``cos_sin_cache`` in ``k_pe``'s dtype, so it serves
    the rows whose table has that dtype.
    """
    write = vllm_op("concat_and_cache_mla_rope_fused")
    dtype_name, divisor = _vllm_cache_args(inputs)
    k_pe = inputs[1]
    no_query = torch.empty((k_pe.shape[0], 0, k_pe.shape[1]), dtype=k_pe.dtype, device=k_pe.device)

    def run(kv_c, k_pe, kv_cache, slots, scale, positions, cos_sin_cache):
        write(
            positions,
            no_query,
            k_pe,
            kv_c,
            cos_sin_cache,
            neox,
            slots,
            _cache_view(kv_cache),
            dtype_name,
            divisor,
        )
        return kv_cache

    # Both the cache and k_pe are written in place.
    return private_inputs(run, inputs, 1, 2)


def _vllm_unfused_rope_write(inputs, neox):
    """vLLM's path for a float32 cos/sin table: FlashInfer's rotation, then the cache write.

    vLLM's DeepSeek rotary embedding keeps its table in float32 and rotates through
    ``flashinfer.apply_rope_with_cos_sin_cache_inplace``; the cache write that follows is
    ``concat_and_cache_mla``. Both launches are timed. The query passed has no heads.
    """
    rotate = flashinfer_op("apply_rope_with_cos_sin_cache_inplace")
    write = vllm_op("concat_and_cache_mla")
    dtype_name, divisor = _vllm_cache_args(inputs)
    k_pe = inputs[1]
    no_query = torch.empty((k_pe.shape[0], 0), dtype=k_pe.dtype, device=k_pe.device)

    def run(kv_c, k_pe, kv_cache, slots, scale, positions, cos_sin_cache):
        rotate(positions, no_query, k_pe, k_pe.shape[1], cos_sin_cache, neox)
        write(kv_c, k_pe, _cache_view(kv_cache), slots, dtype_name, divisor)
        return kv_cache

    return private_inputs(run, inputs, 1, 2)


def _flashinfer_write(inputs):
    """FlashInfer's ``append_paged_mla_kv_cache`` into the two column ranges of the cache.

    Its latent and rope caches are the ``[..., :DC]`` and ``[..., DC:]`` views of the one
    cache, which it writes through their strides. Each slot is one request's single page and
    append position; that request metadata is prepared outside timing.
    """
    write = flashinfer_op("append_paged_mla_kv_cache")
    slots, latent = inputs[3], inputs[0].shape[1]
    page_size = inputs[2].shape[1]
    requests = torch.arange(slots.numel(), dtype=torch.int32, device=slots.device)
    indptr = torch.arange(slots.numel() + 1, dtype=torch.int32, device=slots.device)
    pages = (slots // page_size).to(torch.int32)
    offsets = (slots % page_size).to(torch.int32)

    def run(kv_c, k_pe, kv_cache, slots, scale, positions, cos_sin_cache):
        write(
            kv_c,
            k_pe,
            requests,
            offsets,
            kv_cache[..., :latent],
            kv_cache[..., latent:],
            pages,
            indptr,
            offsets + 1,
        )
        return kv_cache

    return private_inputs(run, inputs, 2)


@pytest.mark.parametrize("case", bench.cases(MLAKVCacheWriteFwdOp), ids=lambda case: case.id)
def test_mla_kv_cache_write_bench(case) -> None:
    inputs = case.inputs
    op = MLAKVCacheWriteFwdOp(**case.arguments)
    kv_c, k_pe, kv_cache, slots, _, positions, cos_sin_cache = inputs
    # The reference clones the cache and returns the clone, so it writes nothing it was
    # handed and needs no private copy; it is the oracle for this case.
    implementations = {"torch-ref": case.reference}
    if positions is None:
        implementations[VLLM_TAG] = _vllm_write(inputs)
        # FlashInfer writes 512 + 64 wide 16-bit rows and has no scale or skip input.
        if (
            kv_cache.dtype == kv_c.dtype
            and (kv_c.shape[1], k_pe.shape[1]) == (512, 64)
            and bool((slots >= 0).all())
        ):
            implementations[FLASHINFER_TAG] = _flashinfer_write(inputs)
    else:
        neox = op.rope_layout == "neox"
        if cos_sin_cache.dtype == k_pe.dtype:
            implementations[VLLM_TAG] = _vllm_rope_fused_write(inputs, neox)
        elif cos_sin_cache.dtype == torch.float32:
            implementations[VLLM_TAG] = _vllm_unfused_rope_write(inputs, neox)
        # FlashInfer has no fused rotation in its cache append.
    bench.Runner(op, case).compare(implementations)
