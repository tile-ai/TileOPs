"""V-tile width resolution for kernels feeding ``[*, BV]`` tiles into ``T.gemm``.

WGMMA requires the gemm N extent (columns of the B operand) to be at least
16 and tilelang rejects narrower B operands at compile time, so a narrower
resolved V-tile is a configuration error to reject eagerly, not to clamp.
"""

__all__ = ["GEMM_MIN_N", "resolve_block_v"]

# tilelang's WGMMA lowering rejects B operands narrower than 16 (verified at
# afcebed1 and c7fabc4; TileOPs #1854). Re-check when bumping tilelang.
GEMM_MIN_N = 16


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
            f"{GEMM_MIN_N}, or 0 for no tiling with dim_v >= {GEMM_MIN_N}")
    if dim_v % bv != 0:
        raise ValueError(f"dim_v ({dim_v}) must be divisible by block_v ({bv})")
    return bv
