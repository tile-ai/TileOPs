"""Shared buffer-aliasing helpers for kernels that accept a caller ``out=``.

Kernels that read inputs and write a caller-provided output concurrently must
reject an ``out`` that overlaps an input in memory, while still allowing the
common workspace-slicing pattern (disjoint slices of one pre-allocated buffer).
"""

import torch

__all__ = ["tensors_overlap"]


def tensors_overlap(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """Return whether nonempty tensors on one device have overlapping byte spans.

    Dense tensors, including transposes, are compared exactly. For strided tensors
    with holes, overlapping bounding spans conservatively count as overlap.
    Disjoint slices of one storage and empty tensors do not overlap. DLPack views
    with distinct storage bases are compared by their addresses.

    Args:
        t1: First tensor with nonnegative strides.
        t2: Second tensor with nonnegative strides.

    Returns:
        Whether the tensors' bounding byte spans intersect on the same device.
    """
    # Pointers live in per-device address spaces, so two tensors on different
    # devices can share a numeric data_ptr by coincidence — guard before any
    # interval math.
    if t1.device != t2.device or t1.numel() == 0 or t2.numel() == 0:
        return False
    # DLPack imports can wrap overlapping addresses in different storages.
    # Storage base equality is therefore not a prerequisite for overlap.

    # Byte span from data_ptr is the largest reachable element offset, not
    # numel: a non-contiguous (sliced/transposed) view reaches past
    # numel*element_size, so numel alone would under-estimate the span and miss
    # a real overlap. max_offset = Σ (dim-1)·stride covers strided layouts;
    # for a contiguous tensor it collapses to numel-1.
    def _byte_span(t: torch.Tensor) -> int:
        max_off = sum((s - 1) * st for s, st in zip(t.shape, t.stride(), strict=True))
        return (max_off + 1) * t.element_size()

    t1_start, t1_end = t1.data_ptr(), t1.data_ptr() + _byte_span(t1)
    t2_start, t2_end = t2.data_ptr(), t2.data_ptr() + _byte_span(t2)
    return t1_start < t2_end and t2_start < t1_end
