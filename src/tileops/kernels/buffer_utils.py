"""Shared buffer-aliasing helpers for kernels that accept a caller ``out=``.

Kernels that read inputs and write a caller-provided output concurrently must
reject an ``out`` that overlaps an input in memory, while still allowing the
common workspace-slicing pattern (disjoint slices of one pre-allocated buffer).
"""

import torch

__all__ = ["tensors_overlap"]


def tensors_overlap(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """True if two tensors share storage AND their byte intervals overlap.

    Sharing one storage with disjoint intervals is safe and common — serving
    frameworks (e.g. vLLM) slice inputs, weights, and outputs from a single
    pre-allocated workspace — so storage identity alone must not be rejected.
    """
    # Pointers live in per-device address spaces, so two tensors on different
    # devices can share a numeric data_ptr by coincidence — guard before any
    # interval math.
    if t1.device != t2.device:
        return False
    if t1.untyped_storage().data_ptr() != t2.untyped_storage().data_ptr():
        return False

    # Byte span from data_ptr is the largest reachable element offset, not
    # numel: a non-contiguous (sliced/transposed) view reaches past
    # numel*element_size, so numel alone would under-estimate the span and miss
    # a real overlap. max_offset = Σ (dim-1)·stride covers strided layouts;
    # for a contiguous tensor it collapses to numel-1.
    def _byte_span(t: torch.Tensor) -> int:
        if t.numel() == 0:
            return 0
        max_off = sum((s - 1) * st for s, st in zip(t.shape, t.stride(), strict=True))
        return (max_off + 1) * t.element_size()

    t1_start, t1_end = t1.data_ptr(), t1.data_ptr() + _byte_span(t1)
    t2_start, t2_end = t2.data_ptr(), t2.data_ptr() + _byte_span(t2)
    return t1_start < t2_end and t2_start < t1_end
