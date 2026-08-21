"""Compile-time RoPE loader policy shared by GQA prefill topologies."""

import tilelang.language as T

__all__ = ["make_prefill_rope_policy"]


def make_prefill_rope_policy(
    fuse_rope: bool,
    rotary_dim: int,
    rope_layout: str,
):
    """Return ``(table_cols, paired_dim, rotate)`` TileLang policies.

    Dense, varlen, and paged prefill have different address loaders, but RoPE
    pairing and arithmetic are identical.  Keeping them here prevents the two
    public layouts from drifting between topology-specific kernels.
    """
    if rope_layout not in ("neox", "interleaved"):
        raise ValueError("rope_layout must be 'neox' or 'interleaved'")
    rope_cols = rotary_dim // 2 if fuse_rope else 1

    @T.macro
    def paired_dim(d):
        if rope_layout == "neox":
            return T.if_then_else(
                d < rope_cols,
                d + rope_cols,
                T.if_then_else(d < rotary_dim, d - rope_cols, d),
            )
        return T.if_then_else(
            d < rotary_dim,
            T.if_then_else(d % 2 == 0, d + 1, d - 1),
            d,
        )

    @T.macro
    def rotate(value, paired_value, logical_pos, d, cos_table, sin_table):
        if fuse_rope:
            freq_idx = d % rope_cols if rope_layout == "neox" else d // 2
            c = cos_table[logical_pos, freq_idx]
            s = sin_table[logical_pos, freq_idx]
            negative_half = d < rope_cols if rope_layout == "neox" else d % 2 == 0
            rotated = T.if_then_else(negative_half, -paired_value, paired_value)
            return T.if_then_else(d < rotary_dim, value * c + rotated * s, value)
        return value

    return rope_cols, paired_dim, rotate
