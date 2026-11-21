"""Where a kernel finds a sequence's state: in batch order, or in its slot of a state pool."""

import tilelang.language as T

__all__ = ["state_row"]


def state_row(slots, row):
    """The row of a state tensor that holds *row*'s state: its slot of a pool, else *row*.

    A slot is widened to int64: a pool's flat offsets pass int32 well within one GPU's
    memory. Without a pool the row is returned unchanged, so a kernel that calls this at each
    use site and is built without a pool generates no new local.

    A slot of -1 reads and writes nothing: TileLang guards each global access whose index it
    cannot bound with ``0 <= slot < pool``. ``tests/state_pool_guard.py`` checks it per kernel.
    """
    return row if slots is None else T.cast(slots[row], "int64")
