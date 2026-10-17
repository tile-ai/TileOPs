import torch

from tileops.kernels.tiling import align_up

# The chunk length the chunked prefill programs are built with. The shared-memory plan
# below sizes their tiles from it, so the plan and every program it sizes must agree.
CHUNK_SIZE = 64


def packed_offsets(batch, seq_len, cu_seqlens, device):
    if cu_seqlens is not None:
        return cu_seqlens
    return torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int64, device=device)


def scan_value_tile(value_tile, arch, *, total, num_seqs, value_heads, dim):
    minimum_tokens = 4096
    sparse_heads = 8
    medium_heads = 16
    medium_tokens = 16384
    narrow_value_tile = 16
    long_tokens = 65536
    long_value_tile = 32
    if (
        num_seqs == 1
        and total >= minimum_tokens
        and (
            value_heads <= sparse_heads or (value_heads <= medium_heads and total <= medium_tokens)
        )
        and dim == 128
        and arch == 89
    ):
        return min(value_tile, narrow_value_tile)
    if (
        arch == 89
        and dim == 128
        and num_seqs == 1
        and value_heads == medium_heads
        and medium_tokens < total <= long_tokens
    ):
        return min(value_tile, long_value_tile)
    return value_tile


def _prepare_bytes(dim, element, sequences, lean):
    tiles = 4 * CHUNK_SIZE * dim * element + 4 * CHUNK_SIZE * dim + 4 * CHUNK_SIZE
    inverse = 2 * CHUNK_SIZE * CHUNK_SIZE * element
    if lean:
        inverse = max(0, inverse - CHUNK_SIZE * dim * element)
    return tiles + inverse + align_up(4 * (sequences + 1), 16)


def _scan_bytes(dim, value_tile, element, sequences):
    tiles = dim * value_tile + 3 * CHUNK_SIZE * dim + CHUNK_SIZE * (CHUNK_SIZE + value_tile)
    return tiles * element + align_up(4 * (sequences + 1), 16)


def _plan(dim, element, sequences, budget, arch):
    def fits(size):
        return (align_up(size, 1024) if arch == 90 else size) <= budget

    lean = not fits(_prepare_bytes(dim, element, sequences, lean=False))
    if lean and not fits(_prepare_bytes(dim, element, sequences, lean=True)):
        return None
    for value_tile in (dim, dim // 2):
        if fits(_scan_bytes(dim, value_tile, element, sequences)):
            return lean, value_tile
    return None
