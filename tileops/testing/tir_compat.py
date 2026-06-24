"""Probe for ``tvm.tir`` availability, used to skip cases that still emit ``tir.*`` calls.

``gqa_fwd_fp8`` and ``topk_selector`` still call into ``tvm.tir``, which is unavailable on the
current tilelang stack (apache-tvm-ffi exposes no usable top-level ``tvm.tir``). Until those
kernels migrate to ``tilelang.language`` (``T.*``), smoke/full cases that build them are skipped
via :func:`tir_available`. Follow-up: replace ``tir.call_extern`` / ``tir.*`` with ``T.*``.
"""

from __future__ import annotations


def tir_available() -> bool:
    """Return ``True`` iff ``tvm.tir`` can be imported on the active stack."""
    try:
        from tvm import tir  # noqa: F401
    except ImportError:
        return False
    return True
