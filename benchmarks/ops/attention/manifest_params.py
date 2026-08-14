from collections.abc import Callable, Iterable
from typing import Any

import pytest
import torch


def torch_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def manifest_params(
    workloads: Iterable[dict[str, Any]],
    build_args: Callable[[dict[str, Any]], tuple[Any, ...]],
    *,
    tune: bool = True,
) -> list:
    params = []
    for workload in workloads:
        label = workload.get("label", "manifest")
        marks = ()
        if reason := workload.get("bench_skip_reason"):
            marks = (pytest.mark.skip(reason=reason),)
        for dtype_name in workload["dtypes"]:
            dtype = torch_dtype(dtype_name)
            params.append(
                pytest.param(
                    *build_args(workload),
                    dtype,
                    tune,
                    id=f"{label}-{dtype_name}",
                    marks=marks,
                )
            )
    return params


def mha_qkv_args(workload: dict[str, Any]) -> tuple[int, int, int, int, bool]:
    batch, seq_len, heads, dim = workload["q_shape"]
    return batch, seq_len, heads, dim, workload.get("is_causal", True)


def gqa_fwd_args(
    workload: dict[str, Any],
) -> tuple[int, int, int, int, int, int, bool, float | None, float | None, int, int,
           bool, int | None, str]:
    batch, seq_len, heads, dim = workload["q_shape"]
    _, kv_seq_len, heads_kv, _ = workload["kv_shape"]
    return (
        batch,
        seq_len,
        kv_seq_len,
        heads,
        heads_kv,
        dim,
        workload.get("is_causal", True),
        workload.get("sm_scale"),
        workload.get("softcap"),
        workload.get("window_size_left", -1),
        workload.get("window_size_right", -1),
        workload.get("fuse_rope", False),
        workload.get("rotary_dim"),
        workload.get("rope_layout", "neox"),
    )


def gqa_qkv_args(workload: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    batch, seq_len, heads, dim = workload["q_shape"]
    _, kv_seq_len, heads_kv, _ = workload["kv_shape"]
    if seq_len != kv_seq_len:
        raise ValueError("gqa_fwd_args requires q_shape and kv_shape to share seq_len")
    return batch, seq_len, heads, heads_kv, dim, workload.get("is_causal", True)


def gqa_prefill_varlen_args(
    workload: dict[str, Any],
) -> tuple[int, list[int], list[int], int, int, int, bool, float | None, float | None,
           int, int, bool, int | None, str, torch.dtype | None]:
    return (
        workload["batch"],
        list(workload["q_lens"]),
        list(workload["kv_lens"]),
        workload["heads"],
        workload["heads_kv"],
        workload["dim"],
        workload.get("is_causal", True),
        workload.get("sm_scale"),
        workload.get("softcap"),
        workload.get("window_size_left", -1),
        workload.get("window_size_right", -1),
        workload.get("fuse_rope", False),
        workload.get("rotary_dim"),
        workload.get("rope_layout", "neox"),
        torch_dtype(workload["output_dtype"]) if workload.get("output_dtype") else None,
    )


def gqa_prefill_paged_args(
    workload: dict[str, Any],
) -> tuple[
    int,
    list[int],
    list[int],
    int,
    int,
    int,
    int,
    bool,
    bool,
    int | None,
    str,
    float | None,
    float | None,
    int,
    int,
    bool,
    torch.dtype | None,
    torch.dtype | None,
]:
    batch = workload["batch"]
    q_lens = list(workload.get("q_lens") or [workload["total_q"] // batch] * batch)
    cache_lens = list(
        workload.get("cache_lens")
        or [(workload["physical_tokens"] // batch) - (workload["total_q"] // batch)] * batch
    )
    return (
        batch,
        q_lens,
        cache_lens,
        workload["heads"],
        workload["heads_kv"],
        workload["page_size"],
        workload["dim"],
        workload.get("is_causal", True),
        workload.get("fuse_rope", False),
        workload.get("rotary_dim"),
        workload.get("rope_layout", "neox"),
        workload.get("sm_scale"),
        workload.get("softcap"),
        workload.get("window_size_left", -1),
        workload.get("window_size_right", -1),
        workload.get("append_kv", True),
        torch_dtype(workload["cache_dtype"]) if workload.get("cache_dtype") else None,
        torch_dtype(workload["output_dtype"]) if workload.get("output_dtype") else None,
    )


def mha_decode_args(workload: dict[str, Any]) -> tuple[int, int, int, int, int]:
    batch, seq_len_q, heads, dim = workload["q_shape"]
    _, seq_len_kv, _, _ = workload["kv_shape"]
    return batch, heads, seq_len_q, seq_len_kv, dim


def mha_decode_paged_args(workload: dict[str, Any]) -> tuple[int, int, int, int, int, int, bool]:
    batch, seq_len_q, heads, dim = workload["q_shape"]
    seq_len_kv, _, _ = workload["kv_shape"]
    return (
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        workload["page_size"],
        workload.get("is_causal", False),
    )


def gqa_decode_args(
    workload: dict[str, Any],
) -> tuple[int, int, int, int, int, float | None, float | None]:
    batch, heads, dim = workload["q_shape"]
    _, seq_len_kv, heads_kv, _ = workload["kv_shape"]
    return (
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        workload.get("sm_scale"),
        workload.get("softcap"),
    )


def gqa_decode_paged_args(
    workload: dict[str, Any],
) -> tuple[int, int, int, int, int, int, float | None, float | None]:
    batch, heads, dim = workload["q_shape"]
    seq_len_kv, heads_kv, _ = workload["kv_shape"]
    return (
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        workload["page_size"],
        workload.get("sm_scale"),
        workload.get("softcap"),
    )


def mla_decode_args(workload: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    batch, heads, dim = workload["q_shape"]
    _, seq_len_kv, heads_kv, _ = workload["kv_shape"]
    return batch, heads, heads_kv, seq_len_kv, dim, workload["pe_dim"]


def dsa_decode_args(
    workload: dict[str, Any],
) -> tuple[int, int, int, int, int, int, int, int, int, int, float | None]:
    batch, seq_len_q, heads, q_dim = workload["q_shape"]
    _, seq_len_kv, heads_kv, _ = workload["kv_shape"]
    dim_tail = workload["dim_tail"]
    dim = q_dim - dim_tail
    return (
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        dim_tail,
        workload["topk"],
        workload["stride_kv"],
        heads_kv,
        workload["q_start_index_s"],
        workload.get("sm_scale"),
    )
