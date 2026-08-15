"""Padding and tile-shape arithmetic shared by TileLang kernels."""

__all__ = ["ALIGNMENT", "align_up"]

#: Element count a row is padded to before ``T.copy`` moves it through shared
#: memory: 256 elements, i.e. 512 bytes for fp16/bf16 and 1024 for fp32.
ALIGNMENT = 256


def align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*, which must be positive."""
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((n + alignment - 1) // alignment) * alignment
