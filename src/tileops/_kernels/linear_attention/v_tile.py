"""V-tile width resolution for kernels feeding ``[*, BV]`` tiles into ``T.gemm``.

A V tile below what the gemm accepts is a configuration error to reject
eagerly, not to clamp.
"""

__all__ = ["GEMM_MIN_N", "min_gemm_n", "resolve_block_v"]

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
