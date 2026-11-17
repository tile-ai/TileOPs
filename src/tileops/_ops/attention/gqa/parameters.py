def attention_scale(dim: int, sm_scale: float | None) -> float:
    return dim**-0.5 if sm_scale is None else sm_scale


def score_softcap(softcap: float | None) -> float:
    if softcap is None:
        return 0.0
    if softcap < 0:
        raise ValueError("softcap must be non-negative")
    return softcap


def attention_window(window: int, positions: int) -> int:
    """*window*, or ``-1`` when it hides no key.

    A query's KV position and a key position differ by less than *positions*, so a window of
    at least that many keys is no window; replacing it keeps the kernels' sums of a position
    and a window within int32.
    """
    return -1 if window >= positions else window


def rope_rotary_dim(dim: int, rotary_dim: int | None) -> int:
    rotary_dim = dim if rotary_dim is None else rotary_dim
    if rotary_dim <= 0:
        raise ValueError("rotary_dim must be positive")
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")
    if rotary_dim > dim:
        raise ValueError("rotary_dim must not exceed dim")
    return rotary_dim
