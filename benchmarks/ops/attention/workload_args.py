from dataclasses import dataclass
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


def gqa_dense_decode_args(
    workload: dict[str, Any],
) -> tuple[int, int, int, int, int, float | None, float | None]:
    batch, seq_len_q, heads, dim = workload["q_shape"]
    batch_kv, seq_len_kv, heads_kv, dim_kv = workload["kv_shape"]
    if seq_len_q != 1:
        raise ValueError("a Dense decode workload requires q_shape sequence length 1")
    if batch_kv != batch or dim_kv != dim:
        raise ValueError("q_shape and kv_shape must share batch and head dimension")
    return (
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        workload.get("sm_scale"),
        workload.get("softcap"),
    )


@dataclass(frozen=True)
class GQADensePrefillCase:
    """One Dense prefill row, resolved to concrete arguments.

    ``rotary_dim`` is set exactly on a row that passes the RoPE tables, and
    ``out_dtype`` exactly on one whose Q/K/V are FP8.
    """

    batch: int
    seq_len_q: int
    seq_len_kv: int
    heads: int
    heads_kv: int
    dim: int
    is_causal: bool
    sm_scale: float | None
    softcap: float | None
    rotary_dim: int | None
    rope_layout: str
    dtype: torch.dtype
    out_dtype: torch.dtype | None


def gqa_dense_prefill_args(
    workload: dict[str, Any],
    dtype: torch.dtype,
) -> tuple[GQADensePrefillCase]:
    """One :class:`GQADensePrefillCase` for this row and dtype.

    A row passes the FP8 scales or the RoPE tables by carrying their
    ``*_shape`` keys, and a row whose keys, dtypes and shapes disagree is
    rejected rather than measured under the label it claims.
    """
    batch, seq_len_q, heads, dim = workload["q_shape"]
    batch_kv, seq_len_kv, heads_kv, dim_kv = workload["kv_shape"]
    if seq_len_q < 1 or seq_len_q > seq_len_kv:
        raise ValueError("a Dense prefill workload requires 1 <= seq_len_q <= seq_len_kv")
    if batch_kv != batch or dim_kv != dim:
        raise ValueError("q_shape and kv_shape must share batch and head dimension")

    scale_keys = ("q_scale_shape", "k_scale_shape", "v_scale_shape")
    passes_scales = [key for key in scale_keys if key in workload]
    if passes_scales and len(passes_scales) != len(scale_keys):
        raise ValueError(f"a row passing {passes_scales} must pass all of {list(scale_keys)}")
    # One case per dtype the row lists, so a scale-carrying row lists FP8 alone:
    # its 16-bit case would carry the scale keys and drop the scales.
    is_fp8 = workload["dtypes"] == ["float8_e4m3fn"]
    if bool(passes_scales) != is_fp8:
        raise ValueError(
            "a row passing q/k/v_scale lists float8_e4m3fn alone, and no other row lists it"
        )
    # FP8 has no FP8 output, so an FP8 row names the 16-bit one it produces.
    out_dtype = workload.get("dtype")
    if is_fp8 != (out_dtype is not None):
        raise ValueError("an FP8 row pins the output 'dtype' and a 16-bit row does not")

    rope_keys = ("rope_cos_shape", "rope_sin_shape")
    passes_rope = [key for key in rope_keys if key in workload]
    if passes_rope and len(passes_rope) != len(rope_keys):
        raise ValueError(f"a row passing {passes_rope} must pass both of {list(rope_keys)}")
    rotary_dim = workload.get("rotary_dim")
    if bool(passes_rope) != (workload.get("pos_encoding_mode") == "rope"):
        raise ValueError("a row passes rope_cos/rope_sin exactly when pos_encoding_mode is 'rope'")

    # A declared shape is the shape the case builds; the tables cover exactly
    # this KV extent.
    rope_shape = [seq_len_kv, (dim if rotary_dim is None else rotary_dim) // 2]
    for key, shape in (
        ("q_scale_shape", [batch, heads_kv]),
        ("k_scale_shape", [batch, heads_kv]),
        ("v_scale_shape", [batch, heads_kv]),
        ("rope_cos_shape", rope_shape),
        ("rope_sin_shape", rope_shape),
    ):
        if key in workload and list(workload[key]) != shape:
            raise ValueError(f"{key} must be {shape}, the shape this case builds")

    return (
        GQADensePrefillCase(
            batch=batch,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            is_causal=workload.get("is_causal", True),
            sm_scale=workload.get("sm_scale"),
            softcap=workload.get("softcap"),
            rotary_dim=rotary_dim if passes_rope else None,
            rope_layout=workload.get("rope_layout", "neox"),
            dtype=dtype,
            out_dtype=getattr(torch, out_dtype) if out_dtype is not None else None,
        ),
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
