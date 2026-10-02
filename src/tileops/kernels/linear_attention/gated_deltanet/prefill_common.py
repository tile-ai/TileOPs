# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under the MIT License.
# Adapted and modified for TileOps GatedDeltaNet prefill integration.
"""Gated DeltaNet private cache and metadata helpers."""

import functools
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import tilelang
import tilelang.language as T
import torch


def _tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 256). When the cache is full, the oldest entry will be removed.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache: "OrderedDict[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], tuple[tuple[Any, ...], dict[str, Any], Any]]" = OrderedDict()
    cache_size = 256

    def get_id(x: Any):
        if (type(x) is int) or (type(x) is float) or (type(x) is str):
            return x
        else:
            return id(x)

    def make_identity_key(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[int, ...], tuple[tuple[str, int], ...]]:
        args_key = tuple(get_id(a) for a in args)
        kwargs_key = tuple(sorted((k, get_id(v)) for k, v in kwargs.items()))
        return args_key, kwargs_key

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache, cache_size
        key = make_identity_key(args, kwargs)
        if key in cache:
            cache.move_to_end(key, last=True)
            _, _, cached_result = cache[key]
            return cached_result

        result = fn(*args, **kwargs)
        cache[key] = (args, kwargs, result)
        cache.move_to_end(key, last=True)
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return result

    return wrapper


@tilelang.jit()
def _build_prepare_chunk_offsets_kernel(
    chunk_size,
    block_size,
    dtype,
):
    batch_size_plus_1 = T.dynamic("batch_size_plus_1")
    num_threads = min(max(block_size, 32), 128)

    @T.prim_func
    def prepare_chunk_offsets_kernel(
        cu_seqlens: T.Tensor([batch_size_plus_1], dtype=dtype),
        chunk_offsets: T.Tensor([batch_size_plus_1], dtype=dtype),
    ):
        with T.Kernel(1, threads=num_threads) as (bb,):
            _batch_size = T.alloc_var("int32")
            _batch_size = batch_size_plus_1 - 1

            seqlen_start_fragment = T.alloc_fragment((block_size), dtype=dtype)
            seqlen_end_fragment = T.alloc_fragment((block_size), dtype=dtype)
            chunk_offset_fragment = T.alloc_fragment((block_size), dtype=dtype)

            T.copy(cu_seqlens[: batch_size_plus_1 - 1], seqlen_start_fragment)
            T.copy(cu_seqlens[1:], seqlen_end_fragment)

            for i in T.Parallel(block_size):
                chunk_offset_fragment[i] = seqlen_end_fragment[i] - seqlen_start_fragment[i]
                chunk_offset_fragment[i] = (chunk_offset_fragment[i] + chunk_size - 1) // chunk_size
            T.cumsum(src=chunk_offset_fragment, dim=0)

            chunk_offsets[0] = 0
            T.copy(chunk_offset_fragment, chunk_offsets[1:])

    return prepare_chunk_offsets_kernel


@_tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """The per-sequence prefix sum of chunk counts, on the device.

    The chunk count itself stays on the device: reading it costs a device-to-host
    synchronization, and only a caller that allocates a per-chunk buffer needs it.
    """
    chunk_offsets = torch.empty_like(cu_seqlens)
    prepare_chunk_offsets_kernel = _build_prepare_chunk_offsets_kernel(
        chunk_size=chunk_size,
        block_size=tilelang.next_power_of_2(cu_seqlens.shape[0] - 1),
        dtype=cu_seqlens.dtype,
    )
    prepare_chunk_offsets_kernel(cu_seqlens, chunk_offsets)
    return chunk_offsets


# What the comparator's L2 normalization adds under the square root before taking the
# reciprocal, from `fla.modules.l2norm.l2norm_fwd`. The block solve and the forward both
# form a reciprocal norm and must add the same thing.
L2NORM_EPS: float = 1e-6


def step_size(raw, beta_sigmoid: bool, allow_neg_eigval: bool):
    """The delta-rule step size, from what the op handed the kernel.

    The sigmoid runs in float32 and the result returns to *raw*'s dtype, which is what a
    caller transforming ``beta`` itself would hand every stage. Keeping the wider value
    would give the triangular solve and the recurrence two different step sizes, since the
    solve stores it at the activation dtype and the recurrence in float32.

    Args:
        raw: The value read from ``beta``, already indexed.
        beta_sigmoid: ``beta`` carries raw logits rather than the step size.
        allow_neg_eigval: The transform is ``2 * sigmoid`` rather than ``sigmoid``.

    Returns:
        An expression in *raw*'s dtype, which is *raw* itself where no sigmoid applies.
    """
    if not beta_sigmoid:
        return raw
    scaled = T.sigmoid(T.cast(raw, "float32")) * (2.0 if allow_neg_eigval else 1.0)
    return T.Cast(raw.dtype, scaled)
