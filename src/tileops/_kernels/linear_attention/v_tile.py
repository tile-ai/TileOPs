"""V-tile width resolution and launch widths for kernels feeding tiles into ``T.gemm``.

A V tile below what the gemm accepts is a configuration error to reject
eagerly, not to clamp.
"""

import tilelang.language as T

__all__ = [
    "GEMM_MIN_N",
    "gemm_threads",
    "min_gemm_n",
    "min_gemm_n_threads",
    "resolve_block_v",
    "wgmma_capable",
]

# Narrowest N extent one warp group can take.
GEMM_MIN_N = 16


def min_gemm_n(threads: int) -> int:
    """Return the N extent WGMMA needs from a ``FullRow`` gemm's B operand at *threads*.

    Each warp group of 128 threads takes a share of the extent, and a share below
    ``GEMM_MIN_N`` has no legal layout. Under one whole warp group the gemm does not
    reach WGMMA and this returns 0; a kernel with a floor of its own keeps it, and a
    kernel that assigns warp roles itself is not bound by this at all.

    Args:
        threads: Thread count the kernel launches with.
    """
    return GEMM_MIN_N * (threads // 128)


def min_gemm_n_threads(threads: int, n: int) -> int:
    """Halve *threads*, not below 64, until an N extent of *n* reaches ``min_gemm_n``."""
    while threads > 64 and n < min_gemm_n(threads):
        threads //= 2
    return threads


def wgmma_capable(arch: int, elem: int) -> bool:
    """Whether TileLang lowers a gemm over *elem*-byte float operands to WGMMA on SM *arch*."""
    return arch == 90 and elem == 2


def gemm_threads(threads: int, wgmma: bool, *outputs: tuple[int, int]) -> int:
    """Return the widest launch up to *threads*, halving, whose warps split over every output.

    WGMMA takes a gemm where *wgmma* holds, the launch is whole warp groups and M is at
    least 64; four warps then split M and the rest split N, so M must be a multiple of 64
    and N one of 16 per warp group, as an N-major B operand needs. Elsewhere MMA splits the
    warps as TileLang's default ``Square`` policy chooses, and the split must use every warp
    and give each whole 16-by-8 tiles. One warp splits any output with M a multiple of 16
    and N of 8.

    Args:
        threads: Widest launch to consider, a power-of-two multiple of 32.
        wgmma: Whether the target lowers the program's gemms to WGMMA, as ``wgmma_capable``.
        outputs: ``(M, N)`` output extent of each ``T.gemm`` in the program.
    """
    while threads > 32:
        warps = threads // 32
        for m, n in outputs:
            if wgmma and warps % 4 == 0 and m >= 64:
                if m % 64 or n % (4 * warps):
                    break
                continue
            rows, cols = T.GemmWarpPolicy.Square.compute_warp_partition(m, n, warps)
            if rows * cols != warps or (m // 16) % rows or (n // 8) % cols:
                break
        else:
            return threads
        threads //= 2
    return threads


def resolve_block_v(dim_v: int, block_v: int) -> int:
    """Return the effective V-tile width; ``block_v <= 0`` means no tiling.

    Raises:
        ValueError: if the resolved width is below ``GEMM_MIN_N`` or does not
            divide ``dim_v``.
    """
    bv = dim_v if block_v <= 0 else block_v
    if bv < GEMM_MIN_N:
        raise ValueError(
            f"V-tile width {bv} (dim_v={dim_v}, block_v={block_v}) is below "
            f"the minimum T.gemm N extent ({GEMM_MIN_N}); use block_v >= "
            f"{GEMM_MIN_N}, or 0 for no tiling with dim_v >= {GEMM_MIN_N}"
        )
    if dim_v % bv != 0:
        raise ValueError(f"dim_v ({dim_v}) must be divisible by block_v ({bv})")
    return bv
