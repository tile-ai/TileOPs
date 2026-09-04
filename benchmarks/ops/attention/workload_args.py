from typing import Any

import torch


def mha_qkv_args(workload: dict[str, Any]) -> tuple[int, int, int, int, bool]:
    batch, seq_len, heads, dim = workload["q_shape"]
    return batch, seq_len, heads, dim, workload.get("is_causal", True)


def gqa_qkv_args(workload: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    batch, seq_len, heads, dim = workload["q_shape"]
    _, kv_seq_len, heads_kv, _ = workload["kv_shape"]
    if seq_len != kv_seq_len:
        raise ValueError("gqa_qkv_args requires q_shape and kv_shape to share seq_len")
    return batch, seq_len, heads, heads_kv, dim, workload.get("is_causal", True)


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
    float | None,
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
        workload.get("softcap"),
        getattr(torch, workload["cache_dtype"]) if workload.get("cache_dtype") else None,
    )


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


def gqa_prefill_varlen_args(
    workload: dict[str, Any],
) -> tuple[int, list[int], list[int], int, int, int, bool]:
    batch = workload["batch"]
    q_lens = list(workload.get("q_lens") or [workload["total_q"] // batch] * batch)
    kv_lens = list(workload.get("kv_lens") or [workload["total_kv"] // batch] * batch)
    return (
        batch,
        q_lens,
        kv_lens,
        workload["heads"],
        workload["heads_kv"],
        workload["dim"],
        workload.get("is_causal", True),
    )


def gqa_sliding_window_varlen_args(
    workload: dict[str, Any],
) -> tuple[int, list[int], list[int], int, int, int, bool, int, int]:
    batch = workload["batch"]
    q_lens = list(workload.get("q_lens") or [workload["total_q"] // batch] * batch)
    k_lens = list(workload.get("k_lens") or [workload["total_k"] // batch] * batch)
    return (
        batch,
        q_lens,
        k_lens,
        workload["heads"],
        workload["heads_kv"],
        workload["dim"],
        workload.get("is_causal", True),
        workload.get("window_size_left", -1),
        workload.get("window_size_right", -1),
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
