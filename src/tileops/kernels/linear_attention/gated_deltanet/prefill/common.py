# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under the MIT License.
# Adapted and modified for TileOps GatedDeltaNet prefill integration.
"""Gated DeltaNet private cache, metadata, and GEMM lowering helpers."""

import functools
import os
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import tilelang
import tilelang.language as T
import torch


def _shape_dim(buf: Any, idx: int) -> int:
    region = getattr(buf, "region", None)
    if region is not None:
        extents = [int(r.extent) for r in region]
        while len(extents) > 2 and extents[0] == 1:
            extents.pop(0)
        return extents[idx]
    regions = getattr(buf, "regions", None)
    if regions is not None:
        extents = [int(r.extent) for r in regions]
        while len(extents) > 2 and extents[0] == 1:
            extents.pop(0)
        return extents[idx]
    return int(buf.shape[idx])


def _dtype_of(buf: Any) -> str:
    dtype = getattr(buf, "dtype", None)
    if dtype is not None:
        return str(dtype)
    inner = getattr(buf, "buffer", None)
    if inner is not None:
        inner_dtype = getattr(inner, "dtype", None)
        if inner_dtype is not None:
            return str(inner_dtype)
    return ""


def _read_ptr(buf: Any):
    if hasattr(buf, "access_ptr"):
        return buf.access_ptr("r")
    inner = getattr(buf, "buffer", None)
    region = getattr(buf, "region", None)
    if inner is not None and region is not None:
        return T.address_of(inner[tuple(r.min for r in region)])
    return T.address_of(buf[0, 0])


@T.macro
def _gemm_ss_extern(
    a,
    b,
    c,
    m: int,
    n: int,
    k: int,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    clear_accum: bool = False,
    lda: int = 0,
    ldb: int = 0,
):
    if lda == 0:
        lda = m if transpose_a else k
    if ldb == 0:
        ldb = k if transpose_b else n
    name = (
        f"tl::gemm_ss<{m}, {n}, {k}, 4, 1, "
        f"{int(transpose_a)}, {int(transpose_b)}, {int(clear_accum)}, "
        f"{lda}, {ldb}, 0, 0, true>"
    )
    T.sync_threads()
    T.fence_proxy_async()
    T.call_extern("handle", name, _read_ptr(a), _read_ptr(b), c.data)


@T.macro
def _wgmma_gemm_sync(
    a,
    b,
    c,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    policy: Any = None,
    clear_accum: bool = False,
    num_regs: int = 1,
):
    if policy is None:
        policy = T.GemmWarpPolicy.Square
    T.wgmma_gemm(
        a,
        b,
        c,
        transpose_A=transpose_a,
        transpose_B=transpose_b,
        policy=policy,
        clear_accum=clear_accum,
    )
    T.wait_wgmma(0)
    T.warpgroup_fence_operand(c, num_regs=num_regs)


def _gemm_v1(
    a,
    b,
    c,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: Any = None,
    clear_accum: bool = False,
    k_pack: int = 1,
    mbar: Any = None,
):
    """GDN-local GEMM lowering; never mutates the global TileLang namespace."""
    del k_pack, mbar
    m = _shape_dim(c, 0)
    n = _shape_dim(c, 1)
    k = _shape_dim(a, 0) if transpose_A else _shape_dim(a, 1)
    a_dtype = _dtype_of(a)
    b_dtype = _dtype_of(b)
    mode = os.environ.get(
        "TILEOPS_GDN_PREFILL_GEMM_V1_MODE",
        os.environ.get("FLASHQLA_TL019_GEMM_V1_MODE", "default"),
    )
    if mode == "wgmma" and a_dtype in ("float16", "bfloat16") and a_dtype == b_dtype:
        return _wgmma_gemm_sync(
            a,
            b,
            c,
            transpose_a=transpose_A,
            transpose_b=transpose_B,
            policy=policy,
            clear_accum=clear_accum,
            num_regs=max(1, (m * n) // 128),
        )
    if (
        mode == "legacy"
        and a_dtype == "float16"
        and b_dtype == "float16"
        and m % 64 == 0
        and n in (32, 64, 128)
        and k >= 16
        and k % 16 == 0
    ):
        return _gemm_ss_extern(
            a,
            b,
            c,
            m,
            n,
            k,
            transpose_a=transpose_A,
            transpose_b=transpose_B,
            clear_accum=clear_accum,
        )
    return T.gemm(
        a,
        b,
        c,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
        clear_accum=clear_accum,
    )


def tensor_cache(
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


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    chunk_offsets = torch.empty_like(cu_seqlens)
    prepare_chunk_offsets_kernel = _build_prepare_chunk_offsets_kernel(
        chunk_size=chunk_size,
        block_size=tilelang.next_power_of_2(cu_seqlens.shape[0] - 1),
        dtype=cu_seqlens.dtype,
    )
    prepare_chunk_offsets_kernel(cu_seqlens, chunk_offsets)
    return chunk_offsets, chunk_offsets[-1].item()
