"""Padding and tile-shape arithmetic shared by TileLang kernels."""

import tilelang.language as T

__all__ = ["ALIGNMENT", "align_up", "grouped_tile_id"]

# Element count a row is padded to before ``T.copy`` moves it through shared
# memory: 256 elements, i.e. 512 bytes for fp16/bf16 and 1024 for fp32.
ALIGNMENT = 256


def align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*, which must be positive."""
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((n + alignment - 1) // alignment) * alignment


@T.macro
def grouped_tile_id(flat_id, mt, nt, *, group_size_m: int, num_pid_m: int, num_pid_n: int):
    """Write the grouped-rasterization (m, n) tile coordinates of a flat tile id."""
    gin = T.int32(group_size_m * num_pid_n)
    gid = flat_id // gin
    first_m = gid * T.int32(group_size_m)
    gsize = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
    mt[0] = first_m + (flat_id % gin) % gsize
    nt[0] = (flat_id % gin) // gsize
