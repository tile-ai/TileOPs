# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under the MIT License; see THIRD_PARTY_NOTICES.md for details.
# Adapted and modified for TileOps GatedDeltaNet prefill integration.

from __future__ import annotations

import os
from typing import Any

import tilelang.language as T


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
def _gemm_ss_compat(
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
def _wgmma_gemm_sync_compat(
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


def install_gemm_v1_compat() -> None:
    original_gemm = T.gemm

    def gemm_v1_compat(
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
            return _wgmma_gemm_sync_compat(
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
            return _gemm_ss_compat(
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
        return original_gemm(
            a,
            b,
            c,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            clear_accum=clear_accum,
        )

    T.gemm_v1 = gemm_v1_compat  # type: ignore[attr-defined]
