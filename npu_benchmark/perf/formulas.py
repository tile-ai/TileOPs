"""Roofline formulas for performance upper-bound estimation.

Each function takes an Op instance (with shape/dtype attributes bound during
forward()) and returns (flops, bytes).
"""

from __future__ import annotations

from typing import Callable

_DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
}


def _dtype_itemsize(dtype) -> int:
    if isinstance(dtype, str):
        return _DTYPE_BYTES.get(dtype, 4)
    return _DTYPE_BYTES.get(str(dtype).split(".")[-1], 4)


def topk_selector_roofline(op) -> tuple[int, int]:
    batch = int(op.batch)
    seq_len = int(op.seq_len)
    seq_len_kv = int(op.seq_len_kv)
    kv_group = int(op.kv_group)
    topk = int(op.topk)
    in_elem = _dtype_itemsize(getattr(op, "in_dtype", "float32"))
    out_elem = _dtype_itemsize(getattr(op, "out_dtype", "int32"))
    comparisons = batch * seq_len * kv_group * seq_len_kv
    nbytes = comparisons * in_elem + batch * seq_len * 2 * out_elem
    nbytes += batch * seq_len * kv_group * topk * out_elem
    return int(comparisons), int(nbytes)


ROOFLINE_REGISTRY: dict[str, Callable] = {
    "topk_selector_roofline": topk_selector_roofline,
}
