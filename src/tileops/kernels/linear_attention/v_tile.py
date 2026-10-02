"""V-tile width resolution for kernels feeding ``[*, BV]`` tiles into ``T.gemm``.

The chunked state recurrences pass their V tile as the B operand of a
``FullRow`` gemm, and WGMMA splits that operand's N extent across warp groups:
``min_gemm_n`` is the floor those kernels adopt. A V tile below it is a
configuration error to reject eagerly, not to clamp. A kernel that assigns warp
roles itself, as the gated DeltaNet prefill does, is not bound by it.
"""

__all__ = ["GEMM_MIN_N", "min_gemm_n", "resolve_block_v"]

# Narrowest N extent one warp group can take. tilelang's WGMMA lowering rejects
# a narrower share whatever the thread count.
GEMM_MIN_N = 16


def min_gemm_n(threads: int) -> int:
    """Return the N extent WGMMA needs from a ``FullRow`` gemm's B operand at *threads*.

    Each warp group of 128 threads takes a share of the N extent, and a share
    below ``GEMM_MIN_N`` has no legal layout: tilelang rejects it with "Not a
    canonical GMMA_MN layout". Measured on tilelang 0.1.12 / SM90, fp16 and
    bf16 alike: a 16-column operand builds at 128 threads and fails at 256.
    Under one whole warp group the gemm does not reach WGMMA, so this rule sets
    no floor there and returns 0; a kernel with a floor of its own keeps it.

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
