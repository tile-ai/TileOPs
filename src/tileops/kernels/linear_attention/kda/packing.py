"""The packed view both Kimi Delta Attention programs run on.

Equal-length and packed-varlen inputs differ only in where one sequence ends,
so both kernels take one flat token axis and read the boundaries from metadata.
Building that metadata is shape-only work, cached per launch geometry.
"""

import functools

import torch

__all__ = ["chunk_metadata", "sequence_lengths"]


@functools.lru_cache(maxsize=64)
def chunk_metadata(
    lengths: tuple[int, ...], chunk_size: int, device_index: "int | None"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Where every chunk and every sequence starts, for one packed geometry.

    Returns:
        ``(chunk_bos, chunk_len, seq_bos, seq_len, seq_chunk0)`` as ``int32``
        device tensors: the first two index the chunk-parallel launch, the last
        three the sequential scan.
    """
    device = "cuda" if device_index is None else f"cuda:{device_index}"
    chunk_bos: list[int] = []
    chunk_len: list[int] = []
    seq_bos: list[int] = []
    seq_chunk0: list[int] = []
    start = 0
    for length in lengths:
        seq_bos.append(start)
        seq_chunk0.append(len(chunk_bos))
        for offset in range(0, length, chunk_size):
            chunk_bos.append(start + offset)
            chunk_len.append(min(chunk_size, length - offset))
        start += length
    build = functools.partial(torch.tensor, dtype=torch.int32, device=device)
    return (
        build(chunk_bos),
        build(chunk_len),
        build(seq_bos),
        build(list(lengths)),
        build(seq_chunk0),
    )


# The offsets last read off one device tensor, keyed by its storage and version:
# reading them is a synchronisation, and an inference caller hands the same
# tensor back every step.
_READ_OFFSETS: dict[tuple[int, int, int], tuple[int, ...]] = {}


def sequence_lengths(
    batch: int,
    seq_len: int,
    cu_seqlens: "torch.Tensor | None",
    cu_seqlens_cpu: "torch.Tensor | None",
) -> tuple[int, ...]:
    """The length of every sequence the call packs, as plain numbers.

    Reading them off the device is a synchronisation, so the CPU copy is used
    when the caller supplied one, and a device read is remembered against the
    buffer and version it came from.
    """
    if cu_seqlens is None:
        return (seq_len,) * batch
    if cu_seqlens_cpu is not None:
        offsets = cu_seqlens_cpu.tolist()
    else:
        key = (cu_seqlens.data_ptr(), cu_seqlens.numel(), cu_seqlens._version)
        offsets = _READ_OFFSETS.get(key)
        if offsets is None:
            offsets = tuple(cu_seqlens.tolist())
            if len(_READ_OFFSETS) > 64:
                _READ_OFFSETS.clear()
            _READ_OFFSETS[key] = offsets
    return tuple(int(b) - int(a) for a, b in zip(offsets, offsets[1:], strict=False))
